import unittest
import math
import glob
import numpy as np


class TestBasic(unittest.TestCase):

    def test_hello(self):
        print("Hello")


class TestStatistics(unittest.TestCase):

    def setUp(self):
        self.all_zero = [0, 0, 0, 0, 0, 0]
        self.all_one = [1, 1, 1, 1, 1, 1]
        self.half_one = [0, 0, 0, 1, 1, 1]

    def test_ppv(self):
        from metrics import ppv

        self.assertEqual(ppv(self.all_one, self.all_one), 1)
        self.assertEqual(ppv(self.all_zero, self.all_one), 0)
        self.assertEqual(ppv(self.half_one, self.all_one), 0.5)

    def test_npv(self):
        from metrics import npv
        self.assertEqual(npv(self.all_zero, self.all_zero), 1)
        self.assertTrue(math.isnan(npv(self.all_one, self.all_one)))
        self.assertEqual(npv(self.half_one, self.all_zero), 0.5)

    def test_sensitivity(self):
        from metrics import sensitivity
        self.assertTrue(math.isnan(sensitivity(self.all_zero, self.all_zero)))
        self.assertEqual(sensitivity(self.all_one, self.all_one), 1)
        self.assertEqual(sensitivity(self.half_one, self.all_one), 1)
        self.assertEqual(sensitivity(self.half_one, self.all_zero), 0)

    def test_specificity(self):
        from metrics import specificity
        self.assertEqual(specificity(self.all_zero, self.all_zero), 1)
        self.assertTrue(math.isnan(specificity(self.all_one, self.all_one)))
        self.assertEqual(specificity(self.half_one, self.all_zero), 1)

    def test_balanced_accuracy(self):
        from metrics import balanced_accuracy
        self.assertEqual(balanced_accuracy(self.all_zero, self.all_zero), 1)
        self.assertEqual(balanced_accuracy(self.all_one, self.all_one), 1)
        self.assertEqual(balanced_accuracy(self.half_one, self.all_one), 0.5)

    def test_accuracy(self):
        from metrics import accuracy
        self.assertEqual(accuracy(self.all_zero, self.all_zero), 1)
        self.assertEqual(accuracy(self.all_one, self.all_one), 1)
        self.assertEqual(accuracy(self.half_one, self.all_one), 0.5)

    def test_f1(self):
        from metrics import f1
        self.assertEqual(f1(self.all_zero, self.all_zero), 1)
        self.assertEqual(f1(self.all_one, self.all_one), 1)
        self.assertEqual(f1(self.half_one, self.all_one), 2 / 3)

    def test_mcc(self):
        from metrics import mcc
        self.assertTrue(math.isnan(mcc(self.all_zero, self.all_zero)))
        self.assertTrue(math.isnan(mcc(self.all_one, self.all_one)))
        self.assertTrue(math.isnan(mcc(self.half_one, self.all_one)))

    def test_auc(self):
        from metrics import auc

        true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.assertEqual(auc(true, pred), 1)

        true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        pred = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        self.assertEqual(auc(true, pred), 0)

        true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        pred = [0, 1, 2, 3, 4, 5, 9, 6, 7, 8]

        diff = (auc(true, pred) - 0.85714)
        self.assertTrue(diff > 0 and diff < 0.001)

class TestDataset(unittest.TestCase):

    def setUp(self):

        from dataset import QSARDataset

        dataset = QSARDataset(filepath = "test_data/short.csv",
                              delimiter = ",",
                              curation = None,
                              label = "continuous",
                              label_col = 1,
                              smiles_col = 0,
                              cutoff = 4.5)


        self.dataset = dataset

    def test_to_binary(self):

        self.dataset.to_binary(cutoff = 4.5)

        print(self.dataset._labels)
        print(self.dataset.dataset)

class TestModeling(unittest.TestCase):

    def test_working(self):

        from dataset import QSARDataset

        dataset = QSARDataset(filepath = "test_data/short.csv",
                              delimiter = ",",
                              label = "continuous",
                              label_col = 1,
                              smiles_col = 0,
                              cutoff = 4.5)

        dataset.to_binary(cutoff = 4.5)

        from model import RF
        model = RF(n_estimators = 100)

        fp = dataset.descriptor.calc_morgan(dataset.dataset, count = True)
        print(fp)


        model.fit(fp, dataset._labels["binary"])
        pred = model.predict_probability(fp)
        print(pred)

        from metrics import get_classification_metrics


        pred = [int(x > 0.5) for x in pred]
        true = [int(x) for x in dataset._labels["binary"]]
        stats = get_classification_metrics(true, pred)
        print(stats)


    def test_logp(self):

        from dataset import QSARDataset

        dataset = QSARDataset(filepath = "test_data/logp.tsv",
                              delimiter = "\t",
                              curation = None,
                              label = "continuous",
                              label_col = "Kowwin",
                              smiles_col = "Canonical_QSARr",
                              cutoff = 4.5)

        dataset.to_binary(cutoff = 1)



        from model import RF
        model = RF(n_estimators = 100)


        clean_dataset = dataset.get_dataset()
        clean_labels = dataset.get_labels("binary")
        fp = dataset.descriptor.calc_morgan(clean_dataset, count = True)
        
        import numpy as np
        true_labels = clean_labels

        model.fit(fp, true_labels)
        pred = model.predict_probability(fp)

        from metrics import get_classification_metrics

        from metrics import auc
        print("AUC: ", auc(true_labels, pred))
        pred = np.array([int(x > 0.5) for x in pred], dtype = int)


        import metrics
        stats = get_classification_metrics(true_labels, pred)
        print(stats)


    def test_biowin(self):

        from dataset import QSARDataset

        filename = "test_data/physprop_Biowin.smi"
        print(filename)
        dataset = QSARDataset(filepath = filename,
                              delimiter = ",",
                              label_col = 2,
                              smiles_col = "SMILES")



        from model import RF
        model = RF(n_estimators = 100)


        clean_dataset = dataset.get_dataset()

        fp = dataset.descriptor.calc_morgan(clean_dataset, count = True)

        import numpy as np
        true_labels = dataset.get_labels("binary")

        model.fit(fp, true_labels)
        pred = model.predict_probability(fp)

        from metrics import get_classification_metrics

        from metrics import auc
        print("AUC: ", auc(true_labels, pred))
        pred = [int(x > 0.5) for x in pred]
        stats = get_classification_metrics(true_labels, pred)
        print(stats)



    def test_physprop(self):

        filenames = glob.glob("test_data/*.smi")

        from dataset import QSARDataset

        for filename in filenames:
            print(filename)
            dataset = QSARDataset(filepath = filename,
                                  delimiter = ",",
                                  curation = None,
                                  label_col = 2,
                                  smiles_col = "SMILES")

            if not dataset.has_binary_label():
                dataset.to_binary()

            from model import RF
            model = RF(n_estimators = 100)


            clean_dataset = dataset.get_dataset()

            fp = dataset.descriptor.calc_morgan(clean_dataset, count = True)

            import numpy as np
            true_labels = list(np.array(dataset.get_labels("binary"), dtype = int))

            model.fit(fp, true_labels)
            pred = model.predict_probability(fp)

            from metrics import get_classification_metrics

            from metrics import auc
            print("AUC: ", auc(true_labels, pred))
            pred = [int(x > 0.5) for x in pred]
            stats = get_classification_metrics(true_labels, pred)
            print(stats)

class TestCuration(unittest.TestCase):

    def test_benzene(self):

        from rdkit import Chem
        from curate import curate_mol

        mol = Chem.MolFromSmiles("c1ccccc1")

        curated_mol, history = curate_mol(mol)
        print(history)

        self.assertFalse(history.rejected)
        self.assertTrue(history.passed)

    def test_invalid_atom(self):

        from rdkit import Chem
        from curate import curate_mol

        mol = Chem.MolFromSmiles("c1ccccc1([U])")

        curated_mol, history = curate_mol(mol)
        print(history)

        self.assertFalse(history.passed)
        self.assertTrue(history.rejected)

        self.assertTrue("not in list of allowed atoms" in history)

    def test_mixture(self):

        from rdkit import Chem
        from curate import curate_mol

        mol = Chem.MolFromSmiles("c1ccccc1.O")

        curated_mol, history = curate_mol(mol)
        print(history)

        self.assertFalse(history.rejected)
        self.assertTrue(history.passed)

        self.assertTrue("Detected mixture" in history)



    def test_dataset_curate(self):

        from dataset import QSARDataset

        dataset = QSARDataset(filepath = "test_data/physprop_Biowin.smi",
                              delimiter = ",",
                              curation = None,
                              label_col = 2,
                              smiles_col = "SMILES")
        print(dataset.get_dataset())
        dataset.curate()

        print(dataset.get_dataset())


        failed_df = dataset.dataset[~dataset.dataset["Passed curation"]]
        print(f"Molecules that failed curation: {len(failed_df)}")
        [print(x) for x in failed_df["Curation history"]]

        modified_df = dataset.dataset[dataset.dataset["Curation modified structure"]]
        print(f"\nMolecules that were modified by curation: {len(modified_df)}")
        [print(x) for x in modified_df["Curation history"]]
