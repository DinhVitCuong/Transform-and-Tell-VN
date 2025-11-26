from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.meteor.meteor import Meteor
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm
import py_vncorenlp

def vncore_segment(sentences, vncore):
    segmented_sentences = []
    for sentence in sentences:
        # Handle None or empty
        if sentence is None:
            segmented_sentences.append("")
            continue
        if not isinstance(sentence, str) or not sentence.strip():
            segmented_sentences.append("")
            continue

        try:
            seg_result = vncore.word_segment(sentence)
            if isinstance(seg_result, list):
                # join all segmented sentences into one string
                seg_text = " ".join(seg_result)
            else:
                seg_text = str(seg_result)

            segmented_sentences.append(seg_text)

        except Exception as e:
            print(f"Skipping segmentation, using raw sentence instead: '{sentence}' due to error: {e}")
            segmented_sentences.append(sentence)

    return segmented_sentences
def calculate_bleu_scores(reference_captions, candidate_captions):
    smoothing = SmoothingFunction().method1

    bleu_1_scores = [
        sentence_bleu([ref.split()], cand.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing)
        for ref, cand in tqdm(zip(reference_captions, candidate_captions), total=len(reference_captions), desc="Calculating BLEU-1")
    ]
    bleu_2_scores = [
        sentence_bleu([ref.split()], cand.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        for ref, cand in tqdm(zip(reference_captions, candidate_captions), total=len(reference_captions), desc="Calculating BLEU-2")
    ]
    bleu_3_scores = [
        sentence_bleu([ref.split()], cand.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        for ref, cand in tqdm(zip(reference_captions, candidate_captions), total=len(reference_captions), desc="Calculating BLEU-3")
    ]
    bleu_4_scores = [
        sentence_bleu([ref.split()], cand.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        for ref, cand in tqdm(zip(reference_captions, candidate_captions), total=len(reference_captions), desc="Calculating BLEU-4")
    ]

    BLEU_mean_scores = [
        sum(bleu_1_scores) / len(bleu_1_scores),
        sum(bleu_2_scores) / len(bleu_2_scores),
        sum(bleu_3_scores) / len(bleu_3_scores),
        sum(bleu_4_scores) / len(bleu_4_scores)
    ]
    
    print(f"DONE BLEU")
    return BLEU_mean_scores

def calculate_meteor_scores(reference_captions, candidate_captions):
    meteor_scorer = Meteor()
    meteor_scores = []
    for idx, (ref, cand) in tqdm(enumerate(zip(reference_captions, candidate_captions)), 
                                 total=len(reference_captions), desc="Calculating METEOR"):
            score, _ = meteor_scorer.compute_score({idx: [str(ref)]}, {idx: [str(cand)]})

            meteor_scores.append(score)

    print(f"DONE METEOR")
    return [sum(meteor_scores) / len(meteor_scores)]  

def calculate_rouge_scores(reference_captions, candidate_captions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    rouge_scores = [
        scorer.score(ref, cand) for ref, cand in tqdm(zip(reference_captions, candidate_captions), total=len(reference_captions), desc="Calculating ROUGE")
    ]

    rouge_1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
    rouge_l = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
    print(f"DONE ROUGE")
    return [rouge_1, rouge_l]

def calculate_cider_scores(reference_captions, candidate_captions):
    references = {idx: [ref] for idx, ref in enumerate(reference_captions)}
    candidates = {idx: [cand] for idx, cand in enumerate(candidate_captions)}

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(references, candidates)
    print(f"DONE CIDER")
    return [cider_score]

def evaluate(reference_captions, candidate_captions, vncore):
    print("Starting evaluation...")
    segmented_reference_captions = vncore_segment(reference_captions, vncore)
    segmented_candidate_captions = vncore_segment(candidate_captions, vncore)
    bleu_scores = calculate_bleu_scores(segmented_reference_captions, segmented_candidate_captions)
    meteor_scores = calculate_meteor_scores(segmented_reference_captions, segmented_candidate_captions)
    rouge_scores = calculate_rouge_scores(segmented_reference_captions, segmented_candidate_captions)
    cider_scores = calculate_cider_scores(segmented_reference_captions, segmented_candidate_captions)
    return bleu_scores + meteor_scores + rouge_scores + cider_scores
