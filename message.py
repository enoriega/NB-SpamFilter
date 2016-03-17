''' Class representing an email message '''

import email, os, re
import nltk

stopwords = set(nltk.corpus.stopwords.words('english'))
url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.I)
addr = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", re.I)

class Message(object):
    stemmer = nltk.PorterStemmer()

    def __init__(self, id, subject, text, frm, to):
        self.id = id
        self.subject = subject
        self.text = text
        self.frm = self._extract_address(frm)
        self.to = self._extract_address(to)

    @classmethod
    def _flatten_parts(cls, parts):
        ''' Flattens a tree into a list '''
        ret = []
        if type(parts) == str:
            ret.append(parts)
        elif type(parts) == list:
            for part in parts:
                ret += cls._flatten_parts(part)
        elif parts.get_content_type == 'text/plain':
            ret += parts.get_payload()
        return ret

    @classmethod
    def _extract_address(self, txt):
        ''' Extracts an email address from a a FROM or TO field '''

        # To avoid bugs
        if txt is None:
            txt = ""

        m = addr.search(txt)
        if m is not None:
            return m.group(1)
        else:
            return txt

    @classmethod
    def load(cls, path):
        ''' Loads a message from a file '''
        with open(path) as fp:
            msg = email.message_from_file(fp)

        subject = msg['Subject']
        txt = '\n'.join(
                p for p in cls._flatten_parts(msg.get_payload())
                if type(p) == str
            )

        frm = msg['FROM']
        to = msg['TO']
        id = path.split(os.sep)[-1]

        return Message(id, subject, txt, frm, to)

    @property
    def processedText(self):
        ''' Stems and filters stop words, among other things '''

        try:
            # Concatenate subject and body
            txt = self.subject + '\n' + self.text
            # Replace URLs for a token
            txt = url.sub('-URL-', txt)
            # Tokenize
            tokens = nltk.word_tokenize(txt)
            # Stem and remove stop words
            stems = [self.stemmer.stem(w) for w in tokens if w not in stopwords]
        except:
            stems = []

        return ' '.join(stems)

    def __repr__(self):
        return "FROM:%s\tSUBJECT: %s" % (self.frm, self.subject)
