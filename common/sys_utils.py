
import os
from glob import glob
import os.path as op

def copy(src, dst):
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    else:
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    copy(s, d)
                else:
                    with open(s, "rb") as source_file:
                        with open(d, "wb") as dest_file:
                            contents = source_file.read()
                            dest_file.write(contents)
        else:
            with open(src, "rb") as source_file:
                with open(dst, "wb") as dest_file:
                    contents = source_file.read()
                    dest_file.write(contents)


def copy_repo_folder(src_files, dst_folder):
    dst_files = [op.join(dst_folder, op.basename(f)) for f in src_files]
    for src_f, dst_f in zip(src_files, dst_files):
        # logger.info(f"FROM: {src_f}\nTO:{dst_f}")
        copy(src_f, dst_f)


def copy_repo(dst_repo_p):
    dst_folder = dst_repo_p

    if not op.exists(dst_folder):
        # logger.info("Copying repo")
        src_files = glob("./*.py")
        src_files += [
            './configs',
            './datasets',
            './lib',
            './models',
            './systems',
            './utils',
        ]
        os.makedirs(dst_folder)

        copy_repo_folder(src_files, dst_folder)