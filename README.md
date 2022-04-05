# transformer-network

1. word embedding concept
    - word mapping
    - word to vector
    - Conceptualization of Word Vectors
    - Relationhip between word vectors
2. inner product
3. relative similarity concept
4. 

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
But the key 🔑🗝️ here to notice is that words are **not numbers**! they are not in form of numbers! So, whenever we wanna do modeling of Natural Language, this modeling is achieved by **Algorithms** that potentially can work with numbers! So, what we need to achive is a mapping of each word to a number, once achieved, we can do the analysis! So the way we gonna do this is map or relate every word in our vocabulary to a vector that may be more than 2D! And the idea is when the words are similar, they should be near to each other in this vector space and whenever they are unrelated they gotta be far from each other! And whithin the concept of **Learning**, we will learn the mapping of every word to a vector! And there are many ways to do this!

## Word to Vector(word2vec)

Each of the vectors associated with the given words is often called **Embedding**! So the idea of embedding is to map the words to a vector or to embed a word in a vector space!
<img src="./img/w2vec2.png">

So the idea is basically mapping each and every word to a vector!

<img src="./img/w2vec.png">

So now imagine we have a sentence consists of several words sequentially followed by each other! Then this means, we need to map each word to a vector! So, in the below example, C1 is the vector associated with the first word and so on! The way we learn sucha thing is gonna be discussed later in this article!

<img src="./img/w2vecconcept.png">

But what I want you to achieve from here is that;
1. C(i) is the vector associated with the word(i)! Each of the C(i) is a D-dimensional vector!
2. If two words are similar, then we would expect their associated vectors to be nearby/close to each other!
3. Then we gonna learn those vectors from C(1) to C(n) based upon a large corpus of text! So the idea of this learning concept is to achieve proximity of the words to each other!
4. Once we have achieved this codes, then we would have codebook composed of all the vectors associated with the words!
5. Then once we get such codebook then we are ready to start our analysis process with the associated algorithms! Since now we have numbers(vectors) that are well enough to do the mathematical analysis! Look at the example below;

<img src="./img/w2vecexp.png">

The key 🔑 idea in all of these methods is that, for each word in a given document, we should be anle to **predict the presence of the surrounding words**! So, the idea is that, the words have meaning, that meaning implies at a given word, that should indicate particular other words is present in surronding with high probability! So **Word Vectors** which we gonna learn is meant to preserve that concept!  So with that being said, let's see the **Conceptialization of Meaning of Word Vectors**!

## Conceptualization of Word Vectors
Let's assume the following picture to be a assosiated meaning of the word vector! Then let's consider a 10-dimensional word vector which basically represents the meaning of the word! and each of these 10 dimensions are associated with a particular meaning! With this then we can say;
1. Each of these 10 components are associated with a particular meaning which can be viewed as a "topic"! If so, then each of these topics represents the characteristics of the word!
2. If a given word is aligned with the ith component then we would expect the corresponding value of ith element to be positive becuase it aligned with the word!
3. If a given word is not aligned with the ith component then we would expect the corresponding value of i to be negative becuase it is not aligned with the word!

So just notinaly let's look at a word as an example to better understand the concept! So here, I am assigning the "topics" or "themes" to that 10-dimensional vector which gonna be learnt thro our network! Look at the word "Paris" and the vector associated with it below;
<img src="./img/paris.png">

If you look at the word "politics" since politics have had a significant role in the history of Paris, so it aligns with the "paris" word. Consequently, the value of "politics" component/topic is positive! Conversely, if you look at the word "gender" it has nothing to do with paris so the the value of "geder" component/topic is negative!

So the idea that I want ya to gather from this is that, what that word vector is doing, is represnting component by component(+ if aligned with the word and - if not aligned with the word) is providing the **underlying themantic of the word**! So, what we gonna do next is to take this **underlying themantic of the word** to form a new natural language processing modeling!

## Relationship between word vectors


