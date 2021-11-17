from . import GCNGraphClassifier, GINGraphClassifier, ChebyGIN
try:
    from pytorch_structure2vec.s2v_lib.embedding import EmbedMeanField, EmbedLoopyBP
    from . import S2VClassifier
    s2v_available = True
except:
    print('Failed to import S2V surrogate!')
    s2v_available = False


def get_model_class(model_name):
    """Returns a model class which implements `BaseGraphClassifier`."""
    if model_name == 'gcn':
        model_class = GCNGraphClassifier
    elif model_name == 'gin':
        model_class = GINGraphClassifier
    elif model_name == 'chebygin':
        model_class = ChebyGIN
    elif model_name == 's2v' and s2v_available:
        model_class = S2VClassifier
    else:
        raise ValueError
    return model_class
