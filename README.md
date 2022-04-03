# transformer-network

1. word embedding concept
2. one-hot word embedding
3. inner product
4. relative similarity concept
5. 

# 1. Introduction to word embedding concept

Before deep diving into Transformers we need to get familiar with a basic concepts of word embedding! So, the idea is very simple; Word Embedding means that every word in our vocabualry is gonna be mapped to a vector!

<img src="./img/w2vec.png">

In order to understand the concept of word embedding, let's just think of map of globe! So what I'm showing in the following map is several different squres, triangles, and circles (i.e location). What I wanna convey thro this, is the fact that the reigons which are geographocally close to each other, have similar characteristics and very different characteristics from the ones that are far away from each other! So for example, we expect that the people, culture, lifestyle, etc of the people in Asia (circles) are similar to each other but very different from ones in North America (squres)!

<img src="./img/world.png">

So if we think of it from a math viewpoint by considering the lantitude and longitude lines(2D space), we can come up with the following idea:
- if 2 points have similar lantitude and longitude lines, then we would expect that they are very close to each other!
- Whereas if the associated lantitude and longitudes of 2 points is very different, then we would expect that they are far away from each other!
So, we have a concept of **Similarity** manifested thro **Proximity**!

