import os
import random
import time

class StatusUpdater():
    path = '/mnt/nimble/zicfan/palm/status.txt'
    
    def record(self, msg, agent_id):
        if not os.path.exists("/mnt/nimble/zicfan/palm/"):
            return
        
        duration = random.random() * 30
        print(f'Cooling down for {duration} seconds')
        time.sleep(duration)
        
        agent_id = f"{agent_id:04}"
        log_entry = f"Agent {agent_id}: {msg}\n"
        
        with open(self.path, 'a') as f:
            f.write(log_entry)
            
status_updater = StatusUpdater()