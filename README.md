# transformer-network

1. word embedding concept
2. one-hot word embedding
3. inner product
4. relative similarity concept
5. 

# 1. Introduction to word embedding concept

Before deep diving into Transformers 🤗 we need to get familiar with a basic concepts of word embedding! So, the idea is very simple; Word Embedding means that every word in our vocabualry is gonna be mapped to a vector!

<img src="./img/w2vec.png">

In order to understand the concept of word embedding, let's just think of map of globe 🌎! So what I'm showing in the following map is several different squres🟥, triangles🔺, and circles 🟣 (i.e location 📍). What I wanna convey thro this, is the fact that the reigons which are geographocally close to each other, have similar characteristics and very different characteristics from the ones that are far away from each other! So for example, we expect that the people, culture, lifestyle, etc of the people in Asia (circles 🟣🟣) are similar to each other but very different from ones in North America (squres 🟥🟥)!

<img src="./img/world.png">

So if we think of it from a math viewpoint by considering the lantitude and longitude lines(2D space), we can come up with the following idea:
- if 2 points have similar lantitude and longitude lines, then we would expect that they are very close to each other!
- Whereas if the associated lantitude and longitudes of 2 points is very different, then we would expect that they are far away from each other!
So, we have a concept of **Similarity** manifested thro **Proximity**!

So we will think and look similarily to the Word Embedding concept! So 👇;
1. So every vocabulary word is gonna be mapped to a point in 2D space!
2. then the closer the two words are, the more **related or synonymous** they are!
3. and the farther the two words are, the more **dis-similar or non-synonymous** we would expect them to be!

<img src="./img/w2vec1.png">

## Word Mapping
So pretty coceptially we can think of it as having a vocabulary of `V` words(i.e. V1, V2, V3,..... V), then think of mapping each word to a 2D space in longitude and latitude! So the way we would like to do this is to **Learn** this 2D vectors of the words in sucha way that if 2 words are similar to each other, we would want their assosiated longitude and latitude to be near to each other and vica versa!
<img src="./img/longtude.png">


So this concept is the very fundumental block that we need to model our natural language! 
But the key here to notice is that words are **not numbers**! they are not in form of numbers! So, whenever we wanna do modeling of Natural Language, this modeling is achieved by **Algorithms** that potentially can work with numbers! So, what we need to achive is a mapping of each word to a number, once achieved, we can do the analysis! So the way we gonna do this is map or relate every word in our vocabulary to a vector that may be more than 2D! And the idea is when the words are similar, they should be near to each other in this vector space and whenever they are unrelated they gotta be far from each other! And whithin the concept of **Learning**, we will learn the mapping of every word to a vector! And there are many ways to do this!

## Word to Vector(word2vec)

Each of the vectors associated with the given words is often called **Embedding**! So the idea of embedding is to map the words to a vector or to embed a word in a vector space!
<img src="./img/w2vec2.png">