import codecs
import sys
import getopt

import xml.etree.ElementTree as ET

from putiq.model_wrapper import VectorModelWrap
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def read_topic_file(filename, parse_notes=False):
    topics = {}
    topic_file_content = ''

    print "Reading input file..."

    with codecs.open(filename) as input_topic_file:
        topic_file_content = input_topic_file.read()

    i = 0

    while True:
        try:
            s, i = topic_file_content.index("<topic>", i), topic_file_content.index("</topic>", i) + 8

            topic_struct = ET.fromstring(topic_file_content[s:i])

            topic_number = int(topic_struct.find("number").text)
            summary = topic_struct.find("summary").text
            description = topic_struct.find("description").text
            note = topic_struct.find("note").text if parse_notes else None

            topics[topic_number] = {'summary': summary, 'description': description, 'note': note}
        except:
            break

    print "Topics parsed"

    return topics


def save_topic_file(filename, topics):
    topic_string = ""

    print "Saving expanded topics..."

    for topic_no in topics:
        topic_struct = ET.Element("topic")

        topic_num = ET.Element("number")
        topic_num.text = str(topic_no)
        topic_struct.append(topic_num)

        topic_summary = ET.Element("summary")
        topic_summary.text = topics[topic_no]['summary']
        topic_struct.append(topic_summary)

        topic_description = ET.Element("description")
        topic_description.text = topics[topic_no]['description']
        topic_struct.append(topic_description)

        if topics[topic_no]['note'] is not None:
            topic_note = ET.Element("note")
            topic_note.text = topics[topic_no]['note']
            topic_struct.append(topic_note)

        topic_string += "\n" + ET.tostring(topic_struct)

    with codecs.open(filename, "w") as topic_save_file:
        topic_save_file.write(topic_string)
        topic_save_file.flush()
        topic_save_file.close()

    print "Topics saved!"


def expand_query(topics, model_path, dims, glove, binary, expand_summary=True, expand_description=True,
                 expand_notes=True, expansion_separator="\n", separator=";", closest_cutoff=3):
    print "Loading model..."

    vsm = VectorModelWrap(model_path, glove, binary, dims, "")

    print "Model loaded"

    en_stopwords = set(stopwords.words('english'))
    word_tokenize = RegexpTokenizer("\w+")

    print "Expanding topics"
    i = 0

    for topic_no in topics:
        if expand_summary:
            terms = map(lambda w: w.lower(), word_tokenize.tokenize(topics[topic_no]['summary']))
            expansion_list = []

            for term in terms:
                if vsm.word_in_model(term) and term not in en_stopwords and not term.isdigit():
                    expansion_list += vsm.most_similar(positive=term, number=closest_cutoff)

            topics[topic_no]['summary'] += expansion_separator + separator.join(map(lambda x: x[0], expansion_list))

        if expand_description:
            terms = map(lambda w: w.lower(), word_tokenize.tokenize(topics[topic_no]['description']))
            expansion_list = []

            for term in terms:
                if vsm.word_in_model(term) and term not in en_stopwords and not term.isdigit():
                    expansion_list += vsm.most_similar(positive=term, number=closest_cutoff)

            topics[topic_no]['description'] += expansion_separator + separator.join(map(lambda x: x[0], expansion_list))

        if expand_notes and topics[topic_no]['note'] is not None:
            terms = map(lambda w: w.lower(), word_tokenize.tokenize(topics[topic_no]['note']))
            expansion_list = []

            for term in terms:
                if vsm.word_in_model(term) and term not in en_stopwords and not term.isdigit():
                    expansion_list += vsm.most_similar(positive=term, number=closest_cutoff)

            topics[topic_no]['note'] += expansion_separator + separator.join(map(lambda x: x[0], expansion_list))

        i += 1
        print "{} topic expanded".format(i)

    print "Topic expansion successful"

    return topics


def process(input_filename, parse_notes, output_filename, model_path, dims, glove, binary, expand_summary,
            expand_description, expand_notes, expansion_separator, separator, closest_cutoff):
    save_topic_file(output_filename, expand_query(read_topic_file(input_filename, parse_notes), model_path, dims, glove,
                                                  binary, expand_summary, expand_description, expand_notes,
                                                  expansion_separator, separator, closest_cutoff))


def main(argv):
    in_path = ""
    out_path = ""
    model_path = ""

    expand_description = True
    expand_summary = True
    expand_notes = True

    parse_notes = False

    glove = False
    binary = False
    dims = 0

    expansion_separator = "\n"
    separator = ";"

    closest_cutoff = 3

    try:
        opts, args = getopt.getopt(argv, "hi:o:nm:d:gbe:s:c:", ["ss", "sd", "sn"])
    except:
        print "pmc2vec.py -i <in_topics> -o <out_topics> -n -m <model_path> -d <dims> [-g] [-b] " +\
              "-e <expansion_separator> -s <separator> -c <cutoff> [--ss] [--sd] [--sn]"
        print "In and out topics in agreed format. -n if input topics contains <notes>."
        print "-d model dimensionality, -b if saved in binary word2vec format, -g if saved in Glove txt format."
        print "--ss, --sd, --sn to skip expanding summary, description and notes"
        print "Default values:"
        print "\texpansion separator: \\n separator: ;"
        print "\tcutoff: 3"
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print "pmc2vec.py -i <in_topics> -o <out_topics> -n -m <model_path> -d <dims> [-g] [-b] " + \
                  "-e <expansion_separator> -s <separator> -c <cutoff> [--ss] [--sd] [--sn]"
            print "In and out topics in agreed format. -n if input topics contains <notes>."
            print "-d model dimensionality, -b if saved in binary word2vec format, -g if saved in Glove txt format."
            print "--ss, --sd, --sn to skip expanding summary, description and notes"
            print "Default values:"
            print "\texpansion separator: \\n separator: ;"
            print "\tcutoff: 3"
            sys.exit(0)
        elif opt == '-i':
            in_path = arg
        elif opt == '-o':
            out_path = arg
        elif opt == '-m':
            model_path = arg
        elif opt == '-n':
            parse_notes = True
        elif opt == '-d':
            dims = int(arg)
        elif opt == '-g':
            glove = True
        elif opt == '-b':
            binary = True
        elif opt == '-e':
            expansion_separator = arg
        elif opt == '-s':
            separator = arg
        elif opt == '-c':
            closest_cutoff = int(arg)
        elif opt == '--ss':
            expand_summary = False
        elif opt == '--sd':
            expand_description = False
        elif opt == '--sn':
            expand_notes = False

    if in_path == '' or out_path == '' or model_path == '' or dims == 0:
        print "MISSING ARGS!"
        print "in {} out {} model {} dims {}".format(in_path, out_path, model_path, dims)
        print "pmc2vec.py -i <in_topics> -o <out_topics> -n -m <model_path> -d <dims> [-g] [-b] " + \
              "-e <expansion_separator> -s <separator> -c <cutoff> [--ss] [--sd] [--sn]"
        print "In and out topics in agreed format. -n if input topics contains <notes>."
        print "-d model dimensionality, -b if saved in binary word2vec format, -g if saved in Glove txt format."
        print "--ss, --sd, --sn to skip expanding summary, description and notes"
        print "Default values:"
        print "\texpansion separator: \\n separator: ;"
        print "\tcutoff: 3"
        sys.exit(0)

    print "Args parsed"

    process(in_path, parse_notes, out_path, model_path, dims, glove, binary, expand_summary, expand_description,
            expand_notes, expansion_separator, separator, closest_cutoff)

if __name__ == '__main__':
    main(sys.argv[1:])
