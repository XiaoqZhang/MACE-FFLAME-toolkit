from fflame.eval import model_eval

if __name__ == "__main__":
    model_eval(
        "model/data/run_test.xyz",
        "model/data/run_test_eval.xyz",
        macemodellist=[
            "model/model1_stagetwo.model",
            "model/model2_stagetwo.model"
        ],
        device="cuda:0"
    )