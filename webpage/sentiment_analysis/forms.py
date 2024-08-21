from django import forms


class Sentiment(forms.Form):
    text = forms.CharField(max_length=100)
