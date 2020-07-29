# rutau
Text augmentation for russian language

# Installation

```
git clone https://github.com/blanchefort/rutau.git
pip install -r requirements.txt
```

# Usage Example

```
from rutau.anaphorate import anaphorate

sentence = 'Василий купил ботинки. Василий Иванович продал ботинки.'
result = anaphorate(text=sentence, sent_splitting='differently', anaph_type=['PER'])

print(result)
```

Result:
```
[{
  'text': 'Василий купил ботинки. Онпродал ботинки.',
  'antecedent': {
    'text': 'Василий',
    'start': 0, 
    'end': 7
   }, 
   'anaphor': {
    'text': 'Он', 
    'start': 23, 
    'end': 25
   }}, 
 {
  'text': 'Онкупил ботинки. Василий Иванович продал ботинки.', 
  'antecedent': {
    'text': 'Василий Иванович', 
    'start': 17, 
    'end': 33
   }, 
    'anaphor': {
      'text': 'Он', 
      'start': 0, 
      'end': 2
}}]
```
