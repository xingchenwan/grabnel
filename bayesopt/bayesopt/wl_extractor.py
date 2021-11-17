from bayesopt.bayesopt.utils import dgl2grakel
from bayesopt.bayesopt.kernels.weisfeiler_lehman import WeisfeilerLehman
from bayesopt.bayesopt.kernels.continuous_wl import ContinuousWeisfeilerLehman
from bayesopt.bayesopt.kernels.vertex_histogram import VertexHistogram
import numpy as np


class WeisfeilerLehmanExtractor:
    def __init__(self, h: int = 1, sparse=False, debug=False, mode='categorical', node_attr='node_attr1'):
        """
        This class extracts the Weisfeiler-Lehman features from graphs and return as np.arrays.
        :param h: the maximum number of Weisfiler-Lehman iterations
        :param sparse: whether allow sparse representations (to be passed to the VertexHistogram base kernel). Allowed
            values: True, False, 'auto'
        :param debug: bool. Whether to display diagnostic information
        :param mode: str. options = 'categorical', 'continuous'. Whether to use the categorical (classical) or the
            continuous WL feature representations.
            The continuous WL feature extractions support continuous node attributes and the presence of both node and
            edge attributes, where as the classical version only supports discrete/categorical node labels.
        """
        self.h = h
        self.base_ker_spec = VertexHistogram, {'sparse': sparse}
        if mode == 'categorical':
            self.wl = WeisfeilerLehman(n_iter=h, base_graph_kernel=self.base_ker_spec)
        elif mode == 'continuous':
            self.wl = ContinuousWeisfeilerLehman(h=h, node_feat_name=node_attr)
        else:
            raise ValueError(f'Unknown mode selection {mode}')
        self.node_attr = node_attr
        self.mode = mode
        self.debug = debug
        # the feature vector of the training set
        self.base_kernels = None

    def fit(self, g_list: list):
        """Fit the WL feature vector on the input list of graphs. These graphs will be considered the "training set".
        g_list: A list of DGL graphs
        """
        if self.wl is not None:
            del self.wl
        if self.mode == 'categorical':
            self.wl = WeisfeilerLehman(n_iter=self.h, base_graph_kernel=self.base_ker_spec)
            train_graphs = dgl2grakel(g_list, self.node_attr)
        else:
            self.wl = ContinuousWeisfeilerLehman(h=self.h, node_feat_name=self.node_attr)
            train_graphs = g_list
        self.wl.fit(train_graphs)
        self.base_kernels = self.wl.X

    def update(self, g_list: list):
        """This function concatenates the stored training feature vector with the new feature vectors in g_list provided.
        Since the new graphs supplied might introduce new WL features, this def also updates the fitted inv_label
        dict of the WL kernels, and that of base VertexHistogram kernel at each WL iteration level."""
        if self.mode == 'categorical':
            if self.base_kernels is None:
                print('The WL kernel is uninitialised. Call the fit method instead')
                self.fit(g_list)
            else:
                eval_graphs = dgl2grakel(g_list)
                transformed_eval_graphs = self.wl.transform_parse(eval_graphs)
                # First update the WL kernel inv_dict with any new features seen by the update
                for level, inv_label in self.wl._inv_labels.items():
                    self.wl._inv_labels[level] = {**self.wl._inv_labels[level], **self.wl._inv_labels_transform[level]}
                for h in range(len(self.base_kernels)):
                    previous_method_call = self.base_kernels[h]._method_calling
                    # Set the method_calling to 3 (eval mode)
                    self.base_kernels[h]._method_calling = 3
                    # Get the full feature vector of the new inputs
                    feature_vector, labels = self.base_kernels[h].parse_input(transformed_eval_graphs[h], return_label=True)
                    # Update the base kernel fitted feature vector
                    n, d = self.base_kernels[h].X.shape
                    if self.debug:
                        print(n, d)
                    # Add trailing zeros to the fitter vector
                    if feature_vector.shape[1] - d >= 0:
                        zeros = np.zeros((n, feature_vector.shape[1] - d))
                        self.base_kernels[h].X = np.hstack((self.base_kernels[h].X, zeros))
                    # Concatenate with the new feature vector
                    # print(self.base_kernels[h].X.shape, feature_vector.shape)
                    self.base_kernels[h].X = np.vstack((self.base_kernels[h].X, feature_vector))
                    # Then update the base kernels with the new labels seen in the new features
                    self.base_kernels[h]._labels = {**self.base_kernels[h]._labels, **labels}
                    # Restore the mode of the base kernels
                    self.base_kernels[h]._method_calling = previous_method_call
        else:
            if self.wl is None or self.wl.X is None:
                print('The WL kernel is uninitialised. Call the fit method instead')
                self.fit(g_list)
            else:
                # eval_graphs = dgl2networkx(g_list, self.node_attr)
                feat_vector = self.wl.parse_input(g_list, train_mode=True)
                self.wl.X = np.vstack((self.wl.X, feat_vector))

    def transform(self, h_list: list):
        """This is used for prediction mode. Similar to update but the new features seen in the h_list graphs will
        not be recorded by the WL or the base (vertex histogram) kernels."""
        if self.mode == 'categorical':
            if self.base_kernels is None:
                raise ValueError("Base kernel is None. Did you call fit or fit_transform first?")
            eval_graphs = dgl2grakel(h_list)
            # Relabelled eval graphs based on the WL list on the inv_labels of the training set
            transformed_eval_graphs = self.wl.transform_parse(eval_graphs)
            feature_vector = None
            for h in range(len(self.base_kernels)):
                previous_method_call = self.base_kernels[h]._method_calling
                # Set the method_calling to 3 (eval mode)
                self.base_kernels[h]._method_calling = 3
                full_feature_vector = self.base_kernels[h].parse_input(transformed_eval_graphs[h])

                # the full feature vector will be in general longer than the existing one due to the new features. in the
                #   transform mode, these new features are discarded (since they do not affect the kernel values)
                if feature_vector is None:
                    feature_vector = full_feature_vector[:, :self.base_kernels[h].X.shape[1]]
                else:
                    feature_vector = np.hstack((feature_vector, full_feature_vector[:, :self.base_kernels[h].X.shape[1]]))
                # Restore the mode of the base kernels
                self.base_kernels[h]._method_calling = previous_method_call
        else:
            if self.wl is None or self.wl.X is None:
                raise ValueError("Base kernel is None. Did you call fit or fit_transform first?")
            # eval_graphs = dgl2networkx(h_list, self.node_attr)
            feature_vector = self.wl.parse_input(h_list, train_mode=False)
        return feature_vector

    def get_train_features(self):
        """Get the WL feature vector of the graphs of which the WLFeatureExtractor is currently fitted"""
        if self.mode == 'categorical':
            if self.base_kernels is None:
                raise ValueError("Base kernel is None. Did you call fit or fit_transform first?")
            feature_vector = None
            for h in range(len(self.base_kernels)):
                if feature_vector is None:
                    feature_vector = self.base_kernels[h].X
                else:
                    feature_vector = np.hstack((feature_vector, self.base_kernels[h].X))
        else:
            feature_vector = np.copy(self.wl.X)
        return feature_vector


if __name__ == '__main__':
    from src.attack.data import Data
    data = Data(dataset_name='IMDB-MULTI', dataset_split=(0.3, 0.3, 0.4))
    l1 = [data.dataset_a[i][0] for i in range(0, 10)]
    l2 = [data.dataset_a[i][0] for i in range(10, 20)]
    l3 = [data.dataset_a[i][0] for i in range(20, 30)]
    wl = WeisfeilerLehmanExtractor(h=1)
    wl.fit(l1)
    for i, g in enumerate(l2):
        wl.update([g])
    print(wl.get_train_features())
    print(wl.transform(l3))