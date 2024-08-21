from django.shortcuts import render, HttpResponse
from flair.data import Sentence
from flair.nn import Classifier
from .forms import Sentiment
from .absa import get_aspect_eval
# Create your views here.


def HomePageView(request):
    if request.method == 'GET':
        form = Sentiment()
        return render(request,'index.html',{'form':form})
    else:
        form = Sentiment(request.POST)
        if form.is_valid():
            subject = form.cleaned_data['text']
            sentence = Sentence(subject)
            # load the NER tagger
            tagger = Classifier.load('sentiment')
            print(subject)
            # run NER over sentence
            tagger.predict(sentence)
            
            sentiment_value = sentence.labels[0].value
            junk,aspect = get_aspect_eval(subject)
            new_form = Sentiment()
            cont = {
                'sentiment': sentiment_value,
                'show' : True,
                'aspect': aspect,
                'form' : new_form
            }
            
            return render(request,'index.html',context=cont)
        else:
            return HttpResponse("Shit didnt work")


def AboutsPageView(request):
    
    return render(request,'abouts.html')