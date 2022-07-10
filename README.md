# data_compression_exam

Ciao Franci, non so come mai Python mi ha smesso di funzionare correttamente.
Comunque ti ho caricato lo script con cui puoi automatizzare il processo. Spero vada.

In weights_conversion and save.py puoi salvarti tutti i pesi, sia come csv, sia come array numpy. Le variabili che puoi cambiare sono dalla riga 88 in poi. Ti chiederei ancora di non usare flat_fallback e integer_only come tipi di conversione, perché richiedono anche un iteratore su un dataset che non ho ancora fatto.

Come modelli puoi usare BERT (la prima volta che chiami lo script con BERT dovrebbe scaricarti il modello) o un modello ricorrente che mi è capitato di usare. Il secondo l'ho messo in modo tale da avere anche un'applicazione su modelli edge, che normalmente è dove si usano questi tipi di compressione lossy.

Sempre per colpa di Python, non sono riuscito a farti un file di requirements. Comunque direi che oltre alle librerie solite, dovrai installare transformers e tensorflow, almeno. Sorry about that.

Se mi viene in mente altro, ti scrivo su Telegram! Nel frattempo che fai girare i tuoi scriptini io finisco gli altri tipi di quantizzazione e vedo se non è troppo complicato fare altri tipi di ottimizzazione usati (in particolare ce n'è uno, il clustering, che c'è scritto che è apprezzabile soprattutto se affiancato a metodi lossless, tipo gzip. Potrebbe essere interessante confrontare).
