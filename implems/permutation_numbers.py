import torch as ch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from tensorflow import keras
import os

import utils


def train_as_they_said(model, trainloader, testloader, loss_fn, acc_fn, base_save_path, epochs=40):
	# optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)
	# optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
	# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

	for e in range(epochs):
		# Train
		running_loss, running_acc = 0.0, 0.0
		num_samples = 0
		model.train()
		iterator = tqdm(trainloader)
		for (x, y) in iterator:
			x, y = x.cuda(), y.cuda()

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(x)[:, 0]
			loss = loss_fn(outputs, y.float())
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			running_acc  += acc_fn(outputs, y)
			num_samples += x.shape[0]

			iterator.set_description("Epoch %d : [Train] Loss: %.5f Accuacy: %.2f" % (e, running_loss / num_samples, 100 * running_acc / num_samples))

		# Validation
		model.eval()
		running_loss, running_acc = 0.0, 0.0
		num_samples = 0
		for (x, y) in testloader:
			x, y = x.cuda(), y.cuda()

			outputs = model(x)[:, 0]
			loss = loss_fn(outputs, y.float())
			running_loss += loss.item()
			running_acc  += acc_fn(outputs, y)
			num_samples += x.shape[0]

		print("[Val] Loss: %.5f Accuacy: %.2f\n" % (running_loss / num_samples, 100 * running_acc / num_samples))
		ch.save(model.state_dict(), os.path.join(base_save_path, str(e+1) + "_" + str(running_acc.item() / num_samples)) + ".pth")

		# scheduler.step()


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='none', help='which dataset to work on (census/mnist/celeba/processed)')
	args = parser.parse_args()
	utils.flash_utils(args)


	if args.dataset == 'census':
		# Census Income dataset
		ci = utils.CensusIncome("./census_data/")

		sex_filter    = lambda df: utils.filter(df, lambda x: x['sex:Female'] == 1, 0.65)
		race_filter   = lambda df: utils.filter(df, lambda x: x['race:White'] == 0,  1.0)
		income_filter = lambda df: utils.filter(df, lambda x: x['income'] == 1, 0.5)

		num_cfs = 100
		for i in range(1, num_cfs + 1):
			# (x_tr, y_tr), (x_te, y_te), _ = ci.load_data()
			(x_tr, y_tr), (x_te, y_te), _ = ci.load_data(income_filter)
			# clf = RandomForestClassifier(max_depth=30, random_state=0, n_jobs=-1)
			clf = MLPClassifier(hidden_layer_sizes=(60, 30, 30), max_iter=200)
			clf.fit(x_tr, y_tr.ravel())
			print("Classifier %d : Train acc %.2f , Test acc %.2f" % (i,
				100 * clf.score(x_tr, y_tr.ravel()),
				100 * clf.score(x_te, y_te.ravel())))

			dump(clf, os.path.join('census_models_mlp_many/income/', str(i)))

	elif args.dataset == 'celeba':
		# CelebA dataset
		model = utils.FaceModel(512, train_feat=True).cuda()
		model = nn.DataParallel(model)

		import torchvision
		from torchvision import transforms
		# path = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/smile_male"
		path = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/smile_old"
		# path = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/smile_attractive"
		# path = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/smile_all"
		batch_size = 512

		train_transform = transforms.Compose([
											# transforms.Resize(220),
											# transforms.CenterCrop(160),
											transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=0.2),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize((0.5), (0.5))])
		test_transform  = transforms.Compose([
											# transforms.Resize(220),
											# transforms.CenterCrop(160),
											transforms.ToTensor(),
											transforms.Normalize((0.5), (0.5))])

		train_set = torchvision.datasets.ImageFolder(path + "/train", transform=train_transform)
		test_set  = torchvision.datasets.ImageFolder(path+ "/test", transform=test_transform)
		trainloader  = ch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
		testloader   = ch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8)

		loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
		acc_fn = lambda outputs, y: ch.sum((y == (outputs >= 0)))

		# base_save_path = "celeba_models/smile_male_vggface_full"
		# base_save_path = "celeba_models/smile_male_vggface_cropped"
		# base_save_path = "celeba_models/smile_old_vggface_cropped"
		# base_save_path = "celeba_models/smile_attractive_vggface_cropped"
		# base_save_path = "/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped"
		# base_save_path = "/u/as9rw/work/fnb/implems/celeba_models/smile_all_vggface_cropped_augs"
		base_save_path = "/u/as9rw/work/fnb/implems/celeba_models/smile_old_vggface_cropped_augs"
		# base_save_path = "celeba_models/smile_male_vggface_cropped_nofeat"
		# base_save_path = "celeba_models/smile_male_webface_cropped"
		train_as_they_said(model, trainloader, testloader, loss_fn, acc_fn, base_save_path, epochs=10)

	elif args.dataset == 'processed':
		# CelebA dataset
		model = utils.FlatFaceModel(512).cuda()
		model = nn.DataParallel(model)

		import torchvision
		path = "/p/adversarialml/as9rw/datasets/celeba_process_vggface1/smile_attractive_vggface"
		# path = "/p/adversarialml/as9rw/datasets/celeba_process_vggface1/smile_attractive"
		# path = "/p/adversarialml/as9rw/datasets/celeba_process_vggface1/smile_male"
		# path = "/p/adversarialml/as9rw/datasets/celeba_process_vggface1/smile_old"
		myloader = lambda x: np.load(x)
		train_set = torchvision.datasets.DatasetFolder(path + "/train", loader=myloader, extensions="npy")
		test_set = torchvision.datasets.DatasetFolder(path+ "/test", loader=myloader, extensions="npy")
		trainloader = ch.utils.data.DataLoader(train_set, batch_size=4096, shuffle=True, pin_memory=True, num_workers=8)
		testloader   = ch.utils.data.DataLoader(test_set, batch_size=4096, shuffle=True, num_workers=8)

		loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
		acc_fn = lambda outputs, y: ch.sum((y == (outputs >= 0)))
		# base_save_path = "celeba_models/smile_male"
		# base_save_path = "celeba_models/smile_old"
		train_as_they_said(model, trainloader, testloader, loss_fn, acc_fn, base_save_path)

	elif args.dataset == 'mnist':
		# MNIST
		(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
		x_tr = x_tr.astype("float32") / 255
		x_te = x_te.astype("float32") / 255

		x_tr = x_tr.reshape(x_tr.shape[0], -1)
		x_te  = x_te.reshape(x_te.shape[0], -1)

		# Brightness Jitter
		brightness = np.random.uniform(0.1, 0.5, size=(x_tr.shape[0],))
		x_tr = x_tr + np.expand_dims(brightness, -1)
		x_tr = np.clip(x_tr, 0, 1)

		clf = MLPClassifier(hidden_layer_sizes=(128, 32, 16), max_iter=40)
		clf.fit(x_tr, y_tr)
		print(clf.score(x_tr, y_tr))
		print(clf.score(x_te, y_te))

		dump(clf, 'mnist_models/brightness_3')
	else:
		raise ValueError("Dataset not supported yet")
