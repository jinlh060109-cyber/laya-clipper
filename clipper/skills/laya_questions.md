---
name: laya-questions
description: >
  How to write the typed questions Laya answers about every candidate clip.
  Adapted from TypeSafe's typesafe-ai skill (MIT, kept unchanged in
  skills/typesafe-ai/SKILL.md), which is written for Jev, to Laya: an open,
  local, non-autoregressive System 1 model with the same question types, a
  512-token window and known weak spots (noul labels, score levels). Built
  into JEV Clipper: the body below is sent, verbatim, to whichever AI
  proposes the clips.
license: MIT
source: https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md
---

# Writing questions for Laya

Laya is a small decision model. It never writes text: for each question it
returns a typed answer with probabilities, in one fast pass per clip. The app
asks it about every candidate clip, then ranks the clips in code by a
weighted sum of the answers. Your questions decide what "a good clip" means
for this video, so write them the way a careful engineer writes a function
signature.

## What Laya is

Laya (convaiinnovations/laya, Apache-2.0) is TypeSafe Jev's open cousin: the
same three question types, answered in one forward pass per clip on the
user's own machine. It is a 421M-parameter encoder (ModernBERT-large; the
multilingual checkpoint, used for non-English speech, is mmBERT-base). Each
option is scored at its own marker token and a softmax runs over that
question's options, so Laya can only ever pick among the options you wrote.
It is used zero-shot here: it was not trained on clips, so it is only as
good as the wording of your questions. Concrete, observable wording is what
moves it.

## What Laya sees

The state for every clip is this JSON, and nothing else:

    {"content_type": "<kind of video>", "clip": {"text": "<the clip's words>"}, "duration": <seconds>}

It hears no audio and sees no picture. It does not see the rest of the
video, your reasons, your categories or the other questions. Refer to the
clip's words as `clip.text` in your instructions.

Question IDs are not sent to Laya. Put the complete meaning in the
instructions and in each level or option.

## The three question types

| Need | Type | Answer | Scored in the ranking |
| --- | --- | --- | --- |
| Whether one condition holds | noul | probability of yes, 0..1 | yes: the probability |
| Which one of a few kinds | choice | one option, with a probability per option | no: it describes, it does not grade |
| A degree along one described dimension | score | expected level, 0..k-1, with a probability per level | yes: level ÷ top level |

Choose by what the answer means. A choice between "tip", "story" and "joke"
labels a clip. A score ranks it. A noul tests a gate.

Prefer noul for what you want to grade. It is Laya's most accurate type
(about 0.86 against 0.72 for score on its own benchmark). The app asks
Laya every noul as a two-option choice, "Yes. <true_means>" against
"No. <false_means>", because on the English checkpoint the bare true/false
labels outweigh the clip; so true_means and false_means are what Laya
actually compares: write them as two concrete, opposite situations. Use a
score only for a real degree with 3-5 clearly different situations.

## Rules that make Laya accurate

1. **One snap judgment per question.** Ask what a knowledgeable viewer
   decides in a second after reading `clip.text`. "Does the tip give a
   concrete step the viewer can try today?" is good. "Evaluate the clip's
   overall quality" is not: split it into its parts.
2. **One dimension per question.** "Funny and surprising" measures two
   things. A clip high on one and low on the other cannot be placed, and
   confidence drops. Ask two questions.
3. **Score levels describe situations, not degrees.** Each level is judged
   on its own: Laya does not see its number or its neighbours. "The joke's
   punchline gets a laugh from a stranger" gives it something to match.
   "Moderately funny", "better than the last level" or "3/5" does not.
   Order levels from worst to best, 3-5 of them, each distinct.
4. **Noul: one condition, high means yes.** "The speaker names a specific
   number, tool or example" is clear. Avoid negations ("free of filler") and
   two conditions at once. Make the boundary sharp ("any", "at least one").
   true_means and false_means must be concrete and mutually exclusive:
   they are the two options Laya picks between.
5. **Choice: cover every case.** Give 2-6 options with short descriptions
   and add a "none of these" or "other" option when a clip may fit none.
   Laya cannot choose an option you left out.
6. **Keep it short: Laya's window is small.** A question and all its levels
   or options share 192 tokens; the clip gets the other 320 (about 220
   words). Instructions under 25 words, each level or option under 15
   words. Longer text is cut off and the last levels become unreadable.
7. **Judge the clip, not the video.** Laya reads one clip at a time; ask
   about what is in `clip.text`, never about what came before or after it.
8. **Do not repeat the fixed layer.** Every clip is already asked how
   likely a stranger stops scrolling, how strong its opening sentence is,
   whether it has a start (the opening sentence introduces its topic) and
   whether it has an end (the closing sentence is a conclusion). You write
   the variable layer: what matters for this kind of video and the
   viewer's goal. For a tutorial, whether the tip is concrete and
   correct-sounding; for comedy, whether the punchline lands; for an
   interview, whether the guest reveals something new; for a goal like
   "the funniest moments", a question that measures exactly that.
9. **Weights are policy.** Give each score and noul question a weight for
   how much it should count against the others (choice questions carry
   none). Your questions share a fifth of the ranking; the fixed layer
   keeps the rest. Put the most weight on the question closest to the viewer's
   goal. Weights are applied in code after Laya answers, so they never
   change what Laya is asked.
10. **Probabilities, not verdicts.** A noul near 0.5 means Laya is torn,
    not "medium". The app marks answers Laya was unsure of and shows every
    answer to the user; an unsure answer is a reason to write the question
    more concretely, not to ask it twice.
11. **Write in the clip's language.** For non-English speech the app uses
    Laya's multilingual checkpoint; questions in the same language as
    `clip.text` read best.

## Example for a cooking tutorial

    {"id": "actionable_step", "type": "score",
     "instructions": "How usable the cooking advice in `clip.text` is for a home cook.",
     "levels": ["No advice, only chatter or reactions.",
                "General advice with no specific step.",
                "One clear step the viewer could copy.",
                "A precise step with amounts, times or temperatures."]}
    {"id": "names_mistake", "type": "noul",
     "instructions": "`clip.text` names a common cooking mistake and how to avoid it.",
     "true_means": "A specific mistake and its fix are both stated.",
     "false_means": "No mistake is named, or no fix is given."}
    {"id": "moment_kind", "type": "choice",
     "instructions": "What `clip.text` mainly is.",
     "choices": [{"key": "technique", "description": "Showing how to do a step."},
                 {"key": "tasting", "description": "Tasting or reacting to the food."},
                 {"key": "story", "description": "A personal story or anecdote."},
                 {"key": "other", "description": "None of these."}]}
