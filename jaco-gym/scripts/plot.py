import numpy as np
import matplotlib.pyplot as plt

reward = np.load("reward.npy")
# each_reward = np.load("each_reward.npy")
step = np.load("step.npy")
reward_timestep = np.load("reward_timestep.npy")
# print(reward)
# print(step)
def cal_average(cal_list):
    average_step = []
    sum = 0
    for i in range(len(cal_list)):
        sum += cal_list[i]
        average_step.append(sum/(i+1))
    return average_step
    
show=reward_timestep[reward_timestep!=-10 ]
show=show[show!=2]
plt.plot(show)
plt.savefig('aa.png')
plt.clf()
plt.xlabel("timestep")
plt.ylabel("reward")
plt.plot(reward[1:])
plt.plot(cal_average(reward[1:]))
plt.plot([0]*len(reward[1:]))
plt.savefig('reward.png')
plt.clf()

# plt.xlabel("timestep")
# plt.ylabel("each reward")
# plt.plot(each_reward)
# plt.plot(cal_average(each_reward))
# plt.savefig('each_reward.png')
# plt.clf()

plt.xlabel("timestep")
plt.ylabel("reward")
plt.plot(reward_timestep)
plt.plot(cal_average(reward_timestep))
plt.plot([0]*len(reward_timestep))
plt.savefig('reward_timestep.png')
plt.clf()

plt.xlabel("timestep")
plt.ylabel("step")
plt.plot(step[1:])
plt.plot(cal_average(step[1:]))
# plt.plot([30]*len(reward[1:]))
# plt.plot([sum(cal_average(step[1:]))/len(cal_average(step[1:]))]*len(cal_average(step[1:])))
plt.savefig('step.png')
plt.clf()
