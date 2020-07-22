import keras
from tensor_models.classification_models.models_factory import ModelsFactory


class KerasModelsFactory(ModelsFactory):

    @staticmethod
    def get_kwargs():
        return {
            'backend': keras.backend,
            'layers': keras.layers,
            'models': keras.models,
            'utils': keras.utils,
        }


Classifiers = KerasModelsFactory()
