from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough,RunnableParallel

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3.8-flash")

prompt1=PromptTemplate(template="Generate short and simple notes for the following {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(template="generate 5 short  question answer  from the following {text}",
        input_variables=['text']
)

prompt3=PromptTemplate(template="merge the provided notes and quiz into a single document \n notes : {notes}. quiz : {quiz}",
)

parser=StrOutputParser()

runnable_chain=RunnableParallel(
    {
        "notes": prompt1 | model | parser,
        "quiz": prompt2 | model | parser
    }
)

final_chain=prompt3 | model | parser

chains=runnable_chain | final_chain

7

9

text= """

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified foy the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy. sparse) sample vectors as input.
However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy. sparse.
csr_matrix (sparse) with dtype=float64.

"""

result=chains.invoke(text)
print(result)