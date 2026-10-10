import unittest
from ds_mentor_agent import route_message, answer


class RoutingTests(unittest.TestCase):
    def test_greeting_does_not_need_model(self):
        route = route_message("Hello!")
        self.assertEqual(route.conversation_mode, "social")
        self.assertFalse(route.requires_model)

    def test_supportive_mode(self):
        route = route_message("I'm frustrated because I don't understand Python")
        self.assertEqual(route.conversation_mode, "supportive")

    def test_happy_mode(self):
        route = route_message("I finally fixed my bug!")
        self.assertEqual(route.conversation_mode, "happy")

    def test_code_intent(self):
        route = route_message("Can you debug this Python function?")
        self.assertEqual(route.intent, "code")

    def test_social_response_needs_no_provider(self):
        response, route, provider = answer("Hi mentor")
        self.assertIn("AI Engineer Mentor", response)
        self.assertEqual(provider, "No model used")


if __name__ == "__main__":
    unittest.main()
