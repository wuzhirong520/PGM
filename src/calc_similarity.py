from dataset import load_dataset, DATASET_INFO
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import random

dataset_names = ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER','ToxCast', 'HIV', 'PCBA', 'Tox21', 'FreeSolv', 'Lipophilicity', 'ESOL']
# dataset_names = ['BACE', 'BBBP', 'ClinTox', 'SIDER','ToxCast', 'Tox21', 'FreeSolv', 'Lipophilicity', 'ESOL']

print("Loading Datasets...")
datasets = [load_dataset(name) for name in dataset_names]
lens = [len(d) for d in datasets]
selected = [random.sample(range(L),640) for L in lens]
print("Dataset Size : ", lens)

print("Loading Molecules...")
mols = [[Chem.MolFromSmiles(d[j][0]) for j in selected[i]] for i,d in enumerate(datasets)]

print("Loading Fingerprints...")
fps = [[AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in m] for m in mols]

print("Calculating Similarity...")
sims = [[0.0 for i in range(12)] for j in range(12)]
for d1 in range(len(dataset_names)):
    for d2 in range(len(dataset_names)):
        if d1==d2 :
            sims[d1][d2]=999990.0
            continue
        sim = 0.0
        for i, fp1 in enumerate(fps[d1]):
            for j, fp2 in enumerate(fps[d2]):
                similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                sim += similarity
                # print(f"Molecule {i} from first file and Molecule {j} from second file similarity: {similarity}")
        sim /= len(fps[d1])*len(fps[d2])
        sims[d1][d2]=sim
        # print(sim)
print(sims)