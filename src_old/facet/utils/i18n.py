
from ..resources.i18n.translations_en import translations

def get_translation(key_path):
    """
    Get translation using dot notation
    Example: get_translation('eeg.import')
    """
    keys = key_path.split('.')
    value = translations
    for key in keys:
        value = value.get(key, key)
    return value
