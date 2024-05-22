import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1")
def action_pos(status): 
    pos, v, ang, va = status
    #print(status)
    if pos <= 0: 
        return 1
    else: 
        return 0 

def action_angle(status): 
    pos, v, ang, va = status
    #print(status)
    if ang > 0: 
        return 1
    else: 
        return 0
    
def build_q_table(all_states,actions):
    q_table=pd.DataFrame(np.zeros((all_states,len(actions))),columns=actions)
    #q_table=pd.DataFrame(np.random.rand(5,2))
    return q_table

def choose_actions(q_table,current_state):
    array_for_actions=q_table.iloc[current_state,:]   #actions 為某row所有動作的q值array
    if(np.random.uniform()>EPSILON or array_for_actions.all()==0):
        specific_action=np.random.choice([0,1]) #just for carpole,10%隨機選動作
    else:
        specific_action=array_for_actions.argmax()  #有90%機率挑q值最高的動作
    return specific_action

EPSILON=1
ALPHA=0.1  #learning rate
GAMMA=0.99


#states
x_space=np.linspace(-4.8,4.8,10)
velocity_space=np.linspace(-4,4,10)
angle_space=np.linspace(-.2095,2095,10)
angle_vel_space=np.linspace(-4,4,10)
all_states=np.array([x_space,velocity_space,angle_space,angle_vel_space])

action_space=np.array([0,1])

#initial a 11x11x11x11x2 matrix
q_table=np.zeros((len(x_space)+1,len(velocity_space)+1,len(angle_space)+1,len(angle_vel_space)+1,len(action_space)))


#print(q_table)
steps = 0
steps_per_episode=[]
episode_list=[]
for episode in range(10000):   #每回合
    episode_list.append(episode)
    observation, info = env.reset(seed=42)  #初始化
    state_x_space=np.digitize(observation[0],x_space)   #把初始化observation的值，分類到區間裡
    state_velocity_space=np.digitize(observation[1],velocity_space)
    state_angle_space=np.digitize(observation[2],angle_space)
    state_angle_vel_space=np.digitize(observation[3],angle_space)


    terminated=False
    truncated=False

    while not terminated:   #每個遊戲開始到結束的過程
        if(np.random.uniform()>EPSILON):
            action=env.action_space.sample()
            observation,reward,terminated,truncated,info=env.step(action)
            #print("use random action")
        else:
            action=np.argmax(q_table[state_x_space,state_velocity_space,state_angle_space,state_angle_vel_space,:])
            observation,reward,terminated,truncated,info=env.step(action)
            #print("use q_table")

            #state-->s2，(預先估計s2的q_value)
        new_state_x_space=np.digitize(observation[0],x_space)   #把初始化observation的值，分類到區間裡
        new_state_velocity_space=np.digitize(observation[1],velocity_space)
        new_state_angle_space=np.digitize(observation[2],angle_space)
        new_state_angle_vel_space=np.digitize(observation[3],angle_space)

        q_estimate=q_table[state_x_space,state_velocity_space,state_angle_space,state_angle_vel_space,action]
        q_real=reward+GAMMA*(np.max(q_table[new_state_x_space,new_state_velocity_space,new_state_angle_space,new_state_angle_vel_space,:]))

        #更新現在的s1的q_value
        q_table[state_x_space,state_velocity_space,state_angle_space,state_angle_vel_space,action]+=ALPHA*(q_real-q_estimate)

        #state1正式結束，更新為state2
        state_x_space=new_state_x_space
        state_velocity_space=new_state_velocity_space
        state_angle_space=new_state_angle_space
        state_angle_vel_space=new_state_angle_vel_space

        if terminated :
            print("Episode finished after {} steps".format(steps))
            steps_per_episode.append(steps)
            observation, info = env.reset(seed=42)
            steps = 0
            terminated=True
        else:
            steps += 1
    EPSILON=max((EPSILON-0.0001),0) #why it need to decrease???
env.close()

plt.plot(episode_list,steps_per_episode)
plt.show()
