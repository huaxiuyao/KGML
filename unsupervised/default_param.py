
import models
default_dict = {'twitter': {'PredictionNet': models.ALBERTClassifier, 'embedding_dim': 128, 
                            'pred_hidden_dim': 240, 'pred_output_dim': 2, 'context_hidden_dim': 120, 
                            'data_dir': 'data/twitter'}
                }