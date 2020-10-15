import matplotlib.pyplot as plt
f = open("log_sample.tsv","r")
datalist=f.readlines()

epoch = []
train_cost = []
train_cost_recons = []
train_cost_temp = []
train_cost_pred = []
valid_cost = []
valid_cost_recons = []
valid_cost_temp = []
valid_cost_pred = []

for data in datalist[1:]:
    a = data.split('\t')
    epoch.append(a[1])
    train_cost.append(float(a[2]))
    train_cost_recons.append(float(a[9]))
    train_cost_temp.append(float(a[10])*0.5)
    train_cost_pred.append(float(a[11])*0.1)
    valid_cost.append(float(a[3]))
    valid_cost_recons.append(float(a[12]))
    valid_cost_temp.append(float(a[13])*0.5)
    valid_cost_pred.append(float(a[14])*0.1)

plt.style.use("grayscale")
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("train cost")
ax1.set_xlabel("epoch")
ax1.set_xlim([0,1000])
ax1.set_ylim([0.01,5.0])
ax1.plot(train_cost,label="total")
ax1.plot(train_cost_recons,label="recons")
ax1.plot(train_cost_temp,label="alpha*temporal")
ax1.plot(train_cost_pred,label="beta*prediction")
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("valid cost")
ax2.set_xlabel("epoch")
ax2.set_xlim([0,1000])
ax2.set_ylim([0.01,5.0])
ax2.plot(train_cost,label="total")
ax2.plot(train_cost_recons,label="recons")
ax2.plot(train_cost_temp,label="alpha*temporal")
ax2.plot(train_cost_pred,label="beta*prediction")
ax2.legend()

plt.savefig("plot_loss_sample.png")