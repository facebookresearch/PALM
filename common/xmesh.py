import trimesh
import os
import os.path as op
import numpy as np
import common.thing as thing


def pts2mesh(points, subdivisions, radius, colors):
    sphere_meshes = []
    for idx, point in enumerate(points):
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        sphere.apply_translation(point)
        vc = colors[idx, :3] if colors is not None else None
        sphere = xmesh(sphere.vertices, sphere.faces, vc=vc, verbose=False)
        sphere_meshes.append(sphere)
    combined_mesh = trimesh.util.concatenate(sphere_meshes)
    return combined_mesh

def color_pointcloud(points, colors, radius=0.001):
    points = thing.thing2np(points).reshape(-1, 3)
    colors = thing.thing2np(colors).reshape(-1, colors.shape[-1])[:, :3]

    mymesh = []
    for v3d, w3d in zip(points, colors):
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        sphere.apply_translation(v3d)
        sphere = xmesh(sphere.vertices, sphere.faces, vc=w3d, verbose=False)
        mymesh.append(sphere)
    mymesh = trimesh.util.concatenate(mymesh)
    return mymesh


class xmesh(trimesh.Trimesh):
    def __init__(
        self,
        vertices=None,
        faces=None,
        vc=None,
        verbose=True,
        radius=0.005,
        subdivisions=1,
        **kwargs,
    ):
        # input as point cloud
        if (
            faces is None
            and vertices is not None
            and not isinstance(vertices, trimesh.Trimesh)
        ):
            vertices = thing.thing2np(vertices)
            vertices = vertices.reshape(-1, 3)
            assert vc is None, "Does not support colorized point cloud"
            combined_mesh = pts2mesh(vertices, subdivisions, radius, colors=None)
            vertices = combined_mesh.vertices
            faces = combined_mesh.faces

        # input as Trimesh
        if isinstance(vertices, trimesh.Trimesh):
            mesh = vertices
            vertices = mesh.vertices
            faces = mesh.faces
            if vc is None and hasattr(mesh, "vertex_colors"):
                vc = mesh.vertex_colors

        if vertices is not None:
            vertices = thing.thing2np(vertices)
            vertices = vertices.reshape(-1, 3)

        if faces is not None:
            faces = thing.thing2np(faces)
            faces = faces.reshape(-1, 3)

        if vc is not None:
            vc = thing.thing2np(vc)
            if vc.max() <= 1.00001:  # tailing 1 is for numerical issue
                vc = (vc * 255.0).astype(np.int64).astype(np.uint8)
            if len(vc.shape) == 1:
                myvc = np.zeros_like(vertices).astype(np.uint8)
                myvc[:] = vc[None, :]
                vc = myvc
            if vc.shape[1] != 3:
                raise ValueError("Vertex colors should have shape (num_verts, 3)")
            if len(vc) != len(vertices):
                raise ValueError(
                    "Number of vertex colors must match number of vertices"
                )
            self.has_color = True
        else:
            self.has_color = False
        # Initialize the mesh with vertex colors
        super().__init__(vertices, faces, vertex_colors=vc, process=False, **kwargs)
        if verbose:
            v_len = 0 if vertices is None else len(vertices)
            f_len = 0 if faces is None else len(faces)
            print("Initialized mesh with {} vertices and {} faces".format(v_len, f_len))

    def load(filename, verbose=False):
        mesh = trimesh.load(filename, process=False)
        mymesh = xmesh(mesh, verbose=verbose)
        return mymesh

    def export(self, out_p, verbose=True):
        out_p = op.abspath(out_p)
        os.makedirs(op.dirname(out_p), exist_ok=True)
        super().export(out_p)
        if verbose:
            print("Exported mesh to {}".format(out_p))

    def add_pts(self, points, radius=0.001, subdivisions=1, n_sample=-1):
        assert (
            not self.has_color
        ), "Only can add points if the base mesh does not have vc."
        points = thing.thing2np(points).reshape(-1, 3)
        vc = None
        if vc is not None:
            vc = thing.thing2np(vc).reshape(-1, vc.shape[-1])

        if n_sample > 0 and n_sample < len(points):
            indices = np.random.choice(len(points), n_sample, replace=False)
            sampled_points = points[indices]
            vc = vc[indices] if vc is not None else vc
        else:
            sampled_points = points
        sphere_meshes = pts2mesh(sampled_points, subdivisions, radius, vc)
        print(f"Added points: {len(sampled_points)}")

        # Concatenate all sphere meshes with the current mesh
        all_meshes = [self, sphere_meshes]
        combined_mesh = trimesh.util.concatenate(all_meshes)
        self.vertices = combined_mesh.vertices
        self.faces = combined_mesh.faces
