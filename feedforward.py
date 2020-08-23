import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle

from encodings_dataset import EncodingsDataset
# from ff_model import FeedforwardNeuralNetModel

#Step 1. Load dataset
train_dataset=EncodingsDataset('encodings_0_5.pickle')
test_dataset=EncodingsDataset('test-encodings.pickle')
print(torch.cuda.is_available())
#Step 2. Make dataset iterable
batch_size=100
iterations=3000
# epochs=int(iterations/(len(train_dataset)/batch_size))

epochs=5
train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

#Step 3. Create model class


# Step 4. Instantiate model class; input size: 128 (encoding length); output size: 10 (no of labels)
# Hidden dimension: in this case 100; it can be any number and in most cases more==better, but
# it may require bigger datasets.

input_dim=128
hidden_dim=100
output_dim=32
class FeedforwardNeuralNetModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes): # hidden_size determines the number of non-linearity dimensions
		super(FeedforwardNeuralNetModel, self).__init__()
		# Linear function
		self.lin1=nn.Linear(input_size, hidden_size)
		# Non-linear function (hidden layer)
		# self.sigmoid=nn.Sigmoid()
		# self.tanh=nn.Tanh()
		self.relu1=nn.ReLU()
		# Linear function (readout layer)
		self.lin2=nn.Linear(hidden_dim, hidden_dim)
		self.relu2=nn.ReLU()
		self.lin3=nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		# Linear function
		out=self.lin1(x)
		# Non-linear function (hidden layer)
		# out=self.sigmoid(out) # Accuracy - 92%
		# out=self.tanh(out) # Accuracy - 95%
		out=self.relu1(out) # Accuracy - 95%
		# Linear function (readout layer)
		out=self.lin2(out)
		out=self.relu2(out)
		out=self.lin3(out)
		return out


model=FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
if torch.cuda.is_available():
	model.cuda()

#Step 5. Instantiate Loss Class
criterion=nn.CrossEntropyLoss()

#Step 6. Instantiate Optimizer Class
learning_rate=.1
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

#Step 7. Train model
iter=0
for epoch in range(epochs):
	for i, (encodings, labels) in enumerate(train_loader):
		if torch.cuda.is_available():
			encodings=Variable(encodings.view(-1, 128)).cuda()
			labels=Variable(labels).cuda()
		else:
			encodings=Variable(encodings.view(-1, 128))
			labels=Variable(labels)

		optimizer.zero_grad()
		outputs=model(encodings.float())
		loss=criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		iter+=1

		if iter%100==0:
			correct=0
			total=0
			for images, labels in test_loader:
				if torch.cuda.is_available():
					images=Variable(images.view(-1, 128)).cuda()
				else:
					images=Variable(images.view(-1, 128))
				outputs=model(images.float())
				_, predicted=torch.max(outputs.data, 1)
				total+=labels.size(0)
				correct+=(predicted.cpu()==labels.cpu()).sum()
			accuracy=100*correct//total
			print('Iteration {}, loss {}, accuracy {}%'.format(iter, loss.data, accuracy))

# path='./feedforward.pth'
# torch.save(model, path)
