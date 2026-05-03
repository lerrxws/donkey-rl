from collections import deque
import random


class ReplayBuffer():
    def __init__(self):
        self.buffer =deque(maxlen=500000)
        self.min_buffer_size=1000
    
    def __len__(self):
        return len(self.buffer)

    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size=32):
        if len(self.buffer)<self.min_buffer_size:
            return None
        return random.sample(self.buffer,batch_size)



if __name__=="__main__":
    replay_buffer=ReplayBuffer()
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    replay_buffer.push([0, 0, 80],0,0.1,[0, 0, 60],False)
    replay_buffer.push([0, 0, 60],0,0.1,[0, 0, 20],False)
    replay_buffer.push([0, 0, 20],0,0.1,[0, 0, 0],True)
    print(replay_buffer.sample())