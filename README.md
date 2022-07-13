# data_compression_exam

In weights_conversion_and_save.py puoi salvarti tutti i pesi, sia come csv, sia come array numpy. 

Per chiamare lo script:
python3 weights_conversion_and_save.py MODEL CONVERSION_METHOD CSV (Optional)

MODEL può essere: 
- BERT
- RNN
- clustered_RNN

CONVERSION_METHOD può essere:
- float32
- float16
- int
- float_fallback
- integer_only

CSV è o True o False, default True

Purtroppo non è possibile fare il modello clusterizzato di BERT