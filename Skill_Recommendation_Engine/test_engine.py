from recommendation_engine import recommend_skills

def test_recommendation():

    result = recommend_skills(0)

    assert len(result) == 3

    print("Test Passed")