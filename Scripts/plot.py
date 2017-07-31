import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import csv

with open('stats.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    stats_full = list(reader)

eval_examples = 256

for s in stats_full:
    train_mode = None
    if s['retrained'] == 'True':
        if s['shallow_retrain']  == 'True':
            train_mode = 'shallow'
        else: 
            train_mode = 'deep'
    else:
        train_mode = 'from_scratch'
    s['train_mode'] = train_mode
    
    s['fps'] = 1.0 / (float(s['eval_time']) / eval_examples)


#Color
N = 6
cmx = cm.rainbow(np.random.rand(3))
c_map = {'shallow' : cmx[0], 'deep' : cmx[1], 'from_scratch': cmx[2]}
cmx = cm.rainbow(range(0, N *50, 50))


lt.figure(figsize=(10, 6))

marker = {'shallow' : 'o', 'deep' : 'h', 'from_scratch' : 's'}
props = dict(boxstyle='round', facecolor='w', alpha=0.7)

for t in c_map.keys():
    stats = list(filter(lambda s: s['train_mode']==t, stats_full))

    colors = [cmx[j] for j in range(N)]
    train_mode = [s['train_mode'] for s in stats]

    #Size
    fps = [float(s['fps'])*20.0 for s in stats]

    #Labels
    names = [s['name'] for s in stats]
    #X
    training_time = [float(s['training_time']) for s in stats]
    #Y
    accuracy = [float(s['accuracy']) for s in stats]

    plt.scatter(training_time, accuracy, s=fps, c=colors,\
                label=t, edgecolor='black', marker=marker[t])

    for i in range(len(stats)):
        print(str(training_time[i]) + ' ' + str(accuracy[i]) + ' ' + names[i] + ' ' + t)
        


plt.xlabel('Training time (s)', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)

plt.legend(loc='lower right', fontsize=12, )

min_fps = min((s['fps'] for s in stats))
max_fps = max((s['fps'] for s in stats))
plt.text(400, 0.55, 'size = fps(%d-%d)' % (max_fps, min_fps), fontsize=14,
        va='bottom', ha='center')

ax = plt.subplot(111)
x = training_time[0]
y = accuracy [0]
ax.plot(x, y, label='alexnet', color=colors[0])
ax.plot(x, y, label='densenet169',color=colors[1])
ax.plot(x, y, label='inception_v3',color=colors[2])
ax.plot(x, y, label='resnet34',color=colors[3])
ax.plot(x, y, label='squeezenet1_1',color=colors[4])
ax.plot(x, y, label='vgg13',color=colors[5])
ax.legend()

plt.title('Transfer learning for Plant Disease classification',y=1.05, fontsize=16)
plt.savefig('results.png')
plt.show()
