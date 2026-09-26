"""
Run: python -m src.test_evaluate

Sanity-checks the F0.5 scorer itself against the challenge's OWN
worked example. Run this BEFORE trusting any threshold-tuning decision
made using evaluate.py -- if the scorer implementation is wrong,
everything downstream (threshold, graph-boost comparison) is silently
miscalibrated. No dataset files needed for this one.
"""
from . import evaluate


def main():
    # Challenge's own worked example (see the problem README):
    # predicted [S2-00047, S2-00193, S3-00812], true [S2-00047, S3-00812]
    # -> precision=2/3, recall=1.0, F0.5 = 0.714
    score = evaluate.score_entity(
        true_matches={"S2-00047", "S3-00812"},
        predicted_matches={"S2-00047", "S2-00193", "S3-00812"},
    )
    expected = 0.714
    print(f"worked-example score: {score:.3f}  (challenge's own example expects ~{expected})")
    assert abs(score - expected) < 0.01, "STOP: scorer does not match the challenge's own example!"
    print("  OK")

    # Singleton handling -- this is where F0.5 macro-averaging gets easy to get wrong.
    assert evaluate.score_entity(set(), set()) == 1.0
    print("correct empty prediction on a true singleton -> 1.0: OK")

    assert evaluate.score_entity(set(), {"S2-00001"}) == 0.0
    print("false positive on a true singleton -> 0.0: OK")

    assert evaluate.score_entity({"S2-00001"}, set()) == 0.0
    print("missed match, predicted empty -> 0.0: OK")

    # Macro average over a tiny hand-built toy case.
    toy_ground_truth = {"S1-A": {"S2-1"}, "S1-B": set(), "S1-C": {"S2-2", "S3-3"}}
    toy_predictions = {"S1-A": {"S2-1"}, "S1-B": set(), "S1-C": {"S2-2"}}
    macro_score = evaluate.macro_f0_5(toy_ground_truth, toy_predictions)
    print(f"\ntoy macro F0.5: {macro_score:.3f}  "
          f"(S1-A=1.0 exact, S1-B=1.0 singleton, S1-C=partial recall -> should print well above 0.5)")

    print("\nAll evaluate.py sanity checks passed.")


if __name__ == "__main__":
    main()
