import subprocess
import gensim

# parameters: $1 min vocab size, $2 vector size, $3 max iter, $4 window size
for dimension in [100, 300]:
    for epoche in [5, 10, 20, 100]: #5, 10, 20, 100
        for learning_rate in [0.01, 0.05, 0.1]: #0.01, 0.05, 0.1
            subprocess.call("./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_%d_%d_%f_esco -dim %d -epoch %d -lr %f -maxn 10 -minCount 1" % (dimension, epoche, learning_rate, dimension, epoche, learning_rate), shell=True)

# for dimension in [100, 300]:
#     for epoche in [5, 10, 20, 100]:
#         for learning_rate in [0.01, 0.05, 0.1]:
#             modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=dimension, epochs=epoche, alpha=learning_rate, mincount=1, max_n=10)
#             str(cbow_%d_%d_%f % (dimension, epoche, learning_rate)) = modello
#             modello.save(/home/maugeri/ojvs-skills/Paper/my_exp/output/)

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=10, epochs=5, alpha=0.01, max_n=10, min_count=3)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_10_5_0.01.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=5, alpha=0.05, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_5_0.05.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=5, alpha=0.1, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_5_0.1.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=10, alpha=0.01, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_10_0.01.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=10, alpha=0.05, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_10_0.05.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=10, alpha=0.1, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_10_0.1.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=20, alpha=0.01, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_20_0.01.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=20, alpha=0.05, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_20_0.05.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=20, alpha=0.1, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_20_0.1.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=100, alpha=0.01, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_100_0.01.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=100, alpha=0.05, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_100_0.05.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=100, epochs=100, alpha=0.1, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_100_0.1.model')




# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=5, alpha=0.01, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_5_0.01.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=5, alpha=0.05, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_5_0.05.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=5, alpha=0.1, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_5_0.1.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=10, alpha=0.01, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_10_0.01.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=10, alpha=0.05, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_10_0.05.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=10, alpha=0.1, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_10_0.1.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=20, alpha=0.01, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_20_0.01.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=20, alpha=0.05, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_20_0.05.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=20, alpha=0.1, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_20_0.1.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=100, alpha=0.01, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_100_0.01.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=100, alpha=0.05, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_100_0.05.model')

# modello = gensim.models.FastText(sentences='/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', vector_size=300, epochs=100, alpha=0.1, max_n=10, min_count=1)
# modello.save('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_100_0.1.model')

# # model = gensim.models.FastText.load('/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_3_1_0.1.model')
# # vettori=model.wv
# # print(vettori[0])


### CODICI DA MANDARE DA SHELL

#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_5_0.01_esco -dim 100 -epoch 5 -lr 0.01 -maxn 10 -minCount 1
##  questo non funziona ./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_5_0.05_esco -dim 100 -epoch 5 -lr 0.05 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_5_0.1_esco -dim 100 -epoch 5 -lr 0.1 -maxn 10 -minCount 1

#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_10_0.01_esco -dim 100 -epoch 10 -lr 0.01 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_10_0.05_esco -dim 100 -epoch 10 -lr 0.05 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_10_0.1_esco -dim 100 -epoch 10 -lr 0.1 -maxn 10 -minCount 1

#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_20_0.01_esco -dim 100 -epoch 20 -lr 0.01 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_20_0.05_esco -dim 100 -epoch 20 -lr 0.05 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_20_0.1_esco -dim 100 -epoch 20 -lr 0.1 -maxn 10 -minCount 1

#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_100_0.01_esco -dim 100 -epoch 100 -lr 0.01 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_100_0.05_esco -dim 100 -epoch 100 -lr 0.05 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_100_100_0.1_esco -dim 100 -epoch 100 -lr 0.1 -maxn 10 -minCount 1





#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_5_0.01_esco -dim 300 -epoch 5 -lr 0.01 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_5_0.05_esco -dim 300 -epoch 5 -lr 0.05 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_5_0.1_esco -dim 300 -epoch 5 -lr 0.1 -maxn 10 -minCount 1

#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_10_0.01_esco -dim 300 -epoch 10 -lr 0.01 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_10_0.05_esco -dim 300 -epoch 10 -lr 0.05 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_10_0.1_esco -dim 300 -epoch 10 -lr 0.1 -maxn 10 -minCount 1

#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_20_0.01_esco -dim 300 -epoch 20 -lr 0.01 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_20_0.05_esco -dim 300 -epoch 20 -lr 0.05 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_20_0.1_esco -dim 300 -epoch 20 -lr 0.1 -maxn 10 -minCount 1

#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_100_0.01_esco -dim 300 -epoch 100 -lr 0.01 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_100_0.05_esco -dim 300 -epoch 100 -lr 0.05 -maxn 10 -minCount 1
#./fastText/fasttext cbow -input Paper/my_exp/dataset/corpus_train_finito.csv -output Paper/my_exp/output/cbow_300_100_0.1_esco -dim 300 -epoch 100 -lr 0.1 -maxn 10 -minCount 1