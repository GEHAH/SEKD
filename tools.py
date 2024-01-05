import torch
import torch.nn as nn
from sklearn import metrics
import pandas as pd
import numpy as np

criterion = nn.CrossEntropyLoss()
def evaluate(model, data_loader):
	model.eval()

	epoch_loss = 0.0
	n = 0
	valid_pred = []
	valid_true = []
	pred_dict = {}

	for data in data_loader:
		with torch.no_grad():
			sequence_name, sequence, label, esm_fs, af_fs, edge_index, CA_coords = data
			y_true, esm_fs, edge_index, CA_coords = label.cuda(), esm_fs.cuda(), edge_index.cuda(), CA_coords.cuda()
			y_true, esm_fs, edge_index, CA_coords = torch.squeeze(y_true).long(), torch.squeeze(
			esm_fs), torch.squeeze(edge_index), torch.squeeze(CA_coords)
		# y_true = F.one_hot(y_true.to(torch.int64)).long().squeeze()
		outputs, outputs_feature = model(esm_fs, CA_coords,
										 edge_index)
		# outputs = model(esm_fs, CA_coords, edge_index)
		output = sum(outputs[:-1]) / len(outputs[:-1])
		loss = criterion(output, y_true)
		softmax = torch.nn.Softmax(dim=1)
		y_pred = softmax(output)
		y_pred = y_pred.cpu().detach().numpy()
		y_true = y_true.cpu().detach().numpy()
		valid_pred += [pred[1] for pred in y_pred]
		valid_true += list(y_true)
		pred_dict[sequence_name[0]] = [pred[1] for pred in y_pred]

		epoch_loss += loss.item()
		n += 1
	epoch_loss_avg = epoch_loss / n

	return epoch_loss_avg, valid_true, valid_pred, pred_dict

def evaluate_single(model, data_loader,threshold):
	model.eval()
	valid_pred = []
	IDs = []
	sequences = []

	threshold_n = threshold/100

	for data in data_loader:
		with torch.no_grad():
			sequence_name, sequence, esm_fs, af_fs, edge_index, CA_coords = data
			esm_fs, edge_index, CA_coords = esm_fs.cuda(), edge_index.cuda(), CA_coords.cuda()
			esm_fs, edge_index, CA_coords = torch.squeeze(esm_fs), torch.squeeze(edge_index), \
											torch.squeeze(CA_coords)
		# y_true = F.one_hot(y_true.to(torch.int64)).long().squeeze()
		outputs, outputs_feature = model(esm_fs, CA_coords,
										 edge_index)
		# outputs = model(esm_fs, CA_coords, edge_index)
		output = sum(outputs[:-1]) / len(outputs[:-1])
		softmax = torch.nn.Softmax(dim=1)
		y_pred = softmax(output)
		y_pred = y_pred.cpu().detach().numpy()
		valid_pred += [pred[1] for pred in y_pred]
		# pred_dict[sequence_name[0]] = [pred[1] for pred in y_pred]

		IDs.append(sequence_name)
		sequences.append(sequence)
	Y_pred = [1 if pred >= threshold_n else 0 for pred in valid_pred]
	data_dic = {'IDS':IDs,'Sequences':sequences,'Labels':str(Y_pred)}

	return Y_pred,data_dic



def analysis(y_true, y_pred, best_threshold=None):
	if best_threshold == None:
		best_f1 = 0
		best_threshold = 0
		for threshold in range(0, 100):
			threshold = threshold / 100
			binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
			binary_true = y_true
			f1 = metrics.f1_score(binary_true, binary_pred)
			# f1 = metrics.roc_auc_score(binary_true, binary_pred)
			if f1 > best_f1:
				best_f1 = f1
				best_threshold = threshold

	binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
	binary_true = y_true

	# binary evaluate
	binary_acc = metrics.accuracy_score(binary_true, binary_pred)
	precision = metrics.precision_score(binary_true, binary_pred)
	recall = metrics.recall_score(binary_true, binary_pred)
	f1 = metrics.f1_score(binary_true, binary_pred)
	AUC = metrics.roc_auc_score(binary_true, y_pred)
	precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
	fpr, tpr, threshold = metrics.roc_curve(binary_true, y_pred)
	aucDot = pd.DataFrame(np.hstack((fpr.reshape((-1, 1)), tpr.reshape((-1, 1)))), columns=['fpr', 'tpr'])
	prcDot = pd.DataFrame(np.hstack((recalls.reshape((-1, 1)), precisions.reshape((-1, 1)))),
						  columns=['recall', 'precision'])
	AUPRC = metrics.auc(recalls, precisions)
	mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

	results = {
		'ACC': binary_acc,
		'PRE': precision,
		'REC': recall,
		'F1': f1,
		'AUC': AUC,
		'AUPRC': AUPRC,
		'MCC': mcc,
		'AUCDot': aucDot,
		'PRCDot': prcDot,
		'threshold': best_threshold
	}
	return results


class Pdb(object):
	""" Object that allows operations with protein files in PDB format. """

	def __init__(self, file_cont=[], pdb_code=""):
		self.cont = []
		self.atom = []
		self.hetatm = []
		self.fileloc = ""
		if isinstance(file_cont, list):
			self.cont = file_cont[:]
		elif isinstance(file_cont, str):
			try:
				with open(file_cont, 'r') as pdb_file:
					self.cont = [row.strip() for row in pdb_file.read().split('\n') if row.strip()]
			except FileNotFoundError as err:
				print(err)

		if self.cont:
			self.atom = [row for row in self.cont if row.startswith('ATOM')]
			self.hetatm = [row for row in self.cont if row.startswith('HETATM')]
			self.conect = [row for row in self.cont if row.startswith('CONECT')]

	def renumber_atoms(self, start=1):
		""" Renumbers atoms in a PDB file. """
		out = list()
		count = start
		for row in self.cont:
			if len(row) > 5:
				if row.startswith(('ATOM', 'HETATM', 'TER', 'ANISOU')):
					num = str(count)
					while len(num) < 5:
						num = ' ' + num
					row = '%s%s%s' % (row[:6], num, row[11:])
					count += 1
			out.append(row)
		return out

	def renumber_residues(self, start=1, reset=False):
		""" Renumbers residues in a PDB file. """
		out = list()
		count = start - 1
		cur_res = ''
		for row in self.cont:
			if len(row) > 25:
				if row.startswith(('ATOM', 'HETATM', 'TER', 'ANISOU')):
					next_res = row[22:27].strip()  # account for letters in res., e.g., '1A'

					if next_res != cur_res:
						count += 1
						cur_res = next_res
					num = str(count)
					while len(num) < 3:
						num = ' ' + num
					new_row = '%s%s' % (row[:23], num)
					while len(new_row) < 29:
						new_row += ' '
					xcoord = row[30:38].strip()
					while len(xcoord) < 9:
						xcoord = ' ' + xcoord
					row = '%s%s%s' % (new_row, xcoord, row[38:])
					if row.startswith('TER') and reset:
						count = start - 1
			out.append(row)
		return out