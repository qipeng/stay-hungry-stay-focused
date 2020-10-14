"""
Generate Qualtrics survey for human evaluation.
"""

from argparse import ArgumentParser
import json
import random
import sys

def total_len(x):
    return sum(len(y['qas']) for y in x)

def main(args):
    with open(args.system1_dump) as f:
        system1 = json.load(f)

    with open(args.system2_dump) as f:
        system2 = json.load(f)

    total_qas = total_len(system1)
    assert total_qas == total_len(system2)

    indices = list(range(total_qas))
    print(f"{len(indices)} total indices to begin with")

    if args.avoid is not None:
        with open(args.avoid) as f:
            for line in f:
                avoidid = int(line.split(',')[0])
                idx = indices.index(avoidid)
                indices = indices[:idx] + indices[idx+1:]

                assert avoidid not in indices

    print(f"{len(indices)} candidates after filtering")

    random.shuffle(indices)
    indices = indices[:args.count]
    indices2id = {x: i for i, x in enumerate(indices)}

    res = [None] * args.count
    record = [None] * args.count
    idx = -1
    for item1, item2 in zip(system1, system2):
        bg = item1['bg']
        bg = bg.replace('<WIKITITLE>', '<span color="#2b8cbe"><strong>').replace('</WIKITITLE>', '</strong></span><br/><br/>\n\n')
        bg = bg.replace('<BG>', '').replace('</BG>', '')

        section_title = item1['section_title'].replace('<SECTITLE>', '').replace('</SECTITLE>', '')

        convo_history = ""

        for qa1, qa2 in zip(item1['qas'], item2['qas']):
            idx += 1
            if idx in indices2id:
                survey_id = indices2id[idx]

                systems = [('human', qa1['question_ref']), (args.system1_dump, qa1['question']), (args.system2_dump, qa2['question'])]
                random.shuffle(systems)

                res[survey_id] = f"""[[Block]]

[[Question:Text]]
{bg}<br/><br/>

<span color="#d7301f"><strong>Section discussed:</strong> {section_title} </span><br/><br/>

{convo_history if len(convo_history) > 0 else "<em>[No conversation history to show]</em><br/>"}<br/>

<strong>Question A:</strong> {systems[0][1].replace('<', '&lt;').replace('>', '&gt;')} <br/>
<strong>Question B:</strong> {systems[1][1].replace('<', '&lt;').replace('>', '&gt;')} <br/>
<strong>Question C:</strong> {systems[2][1].replace('<', '&lt;').replace('>', '&gt;')} <br/>

[[Question:Matrix]]
Rank the <strong>Overall Quality</strong> of each question
[[Choices]]
Question A
Question B
Question C
[[AdvancedAnswers]]
[[Answer]]
1
[[Answer]]
2
[[Answer]]
3

[[Question:Matrix]]
Rank the <strong>Informativeness</strong> of each question
[[Choices]]
Question A
Question B
Question C
[[AdvancedAnswers]]
[[Answer]]
1
[[Answer]]
2
[[Answer]]
3

[[Question:Matrix]]
Rank the <strong>Specificity</strong> of each question
[[Choices]]
Question A
Question B
Question C
[[AdvancedAnswers]]
[[Answer]]
1
[[Answer]]
2
[[Answer]]
3

"""
                record[survey_id] = f"{idx}," + ','.join([x[0] for x in systems])

            convo_history += f"<p style='margin-bottom:10px'><strong>Q:</strong> {qa1['question_ref']}<br/>\n<strong>A:</strong> {qa1.get('answer', '')}</p>"

    with open(args.output_file, 'w') as f:
        print('[[AdvancedFormat]]\n', file=f)
        print('''[[Block]]

[[Question:Text]]

<h2 id="toc_0">Evalutating Questions in a Information-Gathering Conversation</h2>

<p>In this task, you will be asked to read a conversation between two agents on a given topic (an entity from Wikipedia, <em>e.g.</em>, &quot;Albert Einstein&quot;), and evaluate a set of follow-up questions as candidates for the next utterance in the conversation. More specifically, the agents discuss about a given section in that Wikipedia article (<em>e.g.</em>, &quot;Early Life&quot;).</p>

<p>Only one of the two agents, the <strong>teacher</strong>, or answerer, has access to the text of the section, from which answers are provided. The <strong>student</strong>&#39;s (asker&#39;s) goal is to have a meaningful conversation and gather information from this unseen section of text through the conversation.</p>

<h3 id="toc_1">Setting</h3>

<p>You will be provided the same information that is available to the <strong>student</strong>, <em>i.e.</em>, the shared conversational topic (Wikipedia page title, a short introductory paragraph), the section title under discussion, as well as the entire history of conversation between the teacher and the student.</p>

<h3 id="toc_2">Task</h3>

<p>Your task is to evaluate the quality of three candidate questions for each combination of topic under discussion, section title, and conversation history. You will be ranking these questions on three different evaluation metrics, where ties are allowed for any metric (and encouraged if there isn't a clear signal setting candidate questions apart). Specifically, you will be evaluating these questions on their</p>

<ul>
<li><strong>Overall Quality</strong>. A good question should be fluent, specific, and moves the conversation forward. Does this question seem relevant to the conversation? Does it move the conversation forward by gathering more information? Is it grammatical and/or fluent?

<ul>
<li>If you had to choose one of these questions to ask as the student, in which order will you choose these questions (ties are allowed)?</li>
</ul></li>
<li><strong>Informativeness</strong>. A good question in this setting should gather new information that hasn&#39;t already been revealed by the teacher. Does this question attempt to gather new information from the section under discussion?

<ul>
<li>Note that a question doesn&#39;t truly gather new information from the section if references in it are phrased too vaguely to be resolved to anything specific in the conversation history, or if it asks about something completely irrelevant to the (unseen) section under discussion.</li>
<li>Depending on the context, a seemingly repetitive question can actually gather more information (<em>e.g.</em>, asking about other films an actor/actrees has appeared in given the knowledge of some of his/her films). Use your best judgement in these cases.</li>
</ul></li>
<li><strong>Specificity</strong>. A good question should also be tightly related to the topic under discussion, as well as what has just been discussed. Is this question specific for the current conversation, merely applicable to general discussions about this topic, applicable to discussions about virtually any topic, or worse, obviously irrelevant to the current discussion?

<ul>
<li>Note that pronoun use (<em>e.g.</em>, &quot;her&quot;, &quot;it&quot;) shouldn&#39;t be discounted as less specific than mentioning the specific entities, as they are commonly used to refer to topics or entities under discussion.</li>
</ul></li>
</ul>
''', file=f)

        for x in res:
            print(x, file=f)

    with open(args.record_file, 'w') as f:
        for x in record:
            print(x, file=f)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('system1_dump', type=str, help="Prediction dump file of the first system")
    parser.add_argument('system2_dump', type=str, help="Prediction dump file of the second system")
    parser.add_argument('output_file', type=str, default=None, help="Output file for Qualtrics")
    parser.add_argument('record_file', type=str, default=None, help="Records which system is which in the randomized survey, as well as which questions are used")
    parser.add_argument('--count', type=int, default=10, help="Number of survey questions to generate")
    parser.add_argument('--avoid', type=str, default=None, help="Previous record file containing question IDs to avoid")

    args = parser.parse_args()

    main(args)
