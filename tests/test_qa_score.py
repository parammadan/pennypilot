from shoprl.eval.qa_score import correct


def test_triviaqa_alias_in_chatty_reply():
    row = {"set": "triviaqa", "gold": ["arthur ransome"]}
    assert correct(row, "It was Arthur Ransome, the English author.")
    assert not correct(row, "I think it was Charles Dickens.")


def test_triviaqa_whole_word_only():
    assert not correct({"set": "triviaqa", "gold": ["paris"]}, "Comparison shopping!")


def test_mgsm_last_number():
    row = {"set": "mgsm", "gold": ["18"]}
    assert correct(row, "16 - 3 - 4 = 9 eggs, 9 * 2 = $18.")
    assert not correct(row, "She makes 18 dollars... no wait, 20.")
    assert correct({"set": "mgsm", "gold": ["1500"]}, "The answer is 1,500.")
    assert not correct(row, "no number here")
