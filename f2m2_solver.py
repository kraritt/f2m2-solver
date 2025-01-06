import openai
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import re
import pdb  

def LocalFeedback(agent_response):
    """
    Extracts local feedback from an agent's response.
    For simplicity, assume feedback is a summary of the response.
    """
    feedback_prompt = {
        "role": "user",
        "content": f"Please provide your feedback on verifying the facts of {agent_response} to solve the previously provided problem."
    }
    feedback = generate_answer([feedback_prompt])
    feedback_content = feedback.choices[0].message.content
    return feedback_content

def FederatedAggregate(feedbacks):
    """
    Aggregates local feedback from all agents.
    Currently concatenates all feedbacks. 
    Can be enhanced with more sophisticated aggregation methods.
    """
    aggregated_feedback = "\n".join(feedbacks)
    return aggregated_feedback

def BroadcastFeedback(feedback):
    """
    Prepares the feedback to be broadcasted to all agents as a system message.
    """
    broadcast_message = {
        "role": "system",
        "content": f"Aggregated Feedback:\n{feedback}"
    }
    return broadcast_message

client = openai.OpenAI(
  api_key='API_KEY' # Replace with yours
)
def generate_answer(messages):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            n=1,
        )
        return completion
    except Exception as e:
        print(f"Error: {e}. Retrying in 15 seconds...")
        time.sleep(15)
        return generate_answer(messages)

def construct_message(prefix, feedback):
    """
    Constructs the user/system message incorporating aggregated feedback.
    """
    if feedback:
        content = f"{prefix}\n\nAggregated Feedback:\n{feedback}"
    else:
        content = prefix
    return {"role": "user", "content": content}

def construct_assistant_message(completion):
    """
    Extracts the assistant's message from the completion object.
    """
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


def main():
    with open("problem.txt", "r") as file: 
        problem_statement = file.read().strip()

    P_I = f"What is the C++ code to solve {problem_statement}? Make sure to state your answer at the end of the response."
    P_C = f"Can you verify that your solution is correct based on the following responses and feedback? Please reiterate your solution, making sure to state your solution at the end of the response."
    P_R = f"Please provide the final version of your solution based on the following responses and feedback. Please reiterate your solution, making sure to state your solution at the end of the response."
    k = 4  # Number of agents
    N = 3  # Total number of rounds (including initialization and reasoned phases)
    M = 1  # Number of sub-iterations for feedback in each round

    np.random.seed(0)

    # Define agent roles
    agent_roles = ["Philosopher", "Physicist", "Mathematician", "Programmer"]

    # Initialize agent contexts
    agent_contexts = []
    for i in range(k):
        system_role_message = {
            "role": "system",
            "content": (
                f"You are a {agent_roles[i]}. "
                f"Please provide your response inheriting the knowledge having the perspective of this role to solve {problem_statement}."
            ),
        }
        # Initialize with system role and initiation prompt
        agent_contexts.append([system_role_message, {"role": "user", "content": P_I}])

    print("=== Initialization Phase ===\n")
    feedbacks = []
    for i in range(k):
        # Generate initial response
        completion = generate_answer(agent_contexts[i])
        assistant_message = construct_assistant_message(completion)
        agent_contexts[i].append(assistant_message)
        print(f"Agent {i+1} ({agent_roles[i]}) Initial Response:\n{assistant_message['content']}\n")

        # Extract local feedback
        feedback = LocalFeedback(assistant_message["content"])
        feedbacks.append(feedback)
        print(f"Agent {i+1} Feedback:\n{feedback}\n")

    # Federated Aggregate Feedback
    aggregated_feedback = FederatedAggregate(feedbacks)
    print(f"Aggregated Feedback after Initialization:\n{aggregated_feedback}\n")

    # Broadcast Feedback to All Agents
    broadcast_message = BroadcastFeedback(aggregated_feedback)
    for i in range(k):
        agent_contexts[i].append(broadcast_message)

    for t in range(2, N):
        print(f"=== Inter Contest Phase: Round {t} ===\n")
        for m in range(1, M+1):
            print(f"-- Sub-iteration {m} --")
            feedbacks = []
            # Construct the input for this sub-iteration
            for i in range(k):
                # Construct message with current prompt and aggregated feedback 
                user_message = construct_message(P_C, aggregated_feedback)
                agent_contexts[i].append(user_message)

                # Generate response based on updated context
                completion = generate_answer(agent_contexts[i])
                assistant_message = construct_assistant_message(completion)
                agent_contexts[i].append(assistant_message)
                print(f"Agent {i+1} ({agent_roles[i]}) Round {t} Sub-iteration {m} Response:\n{assistant_message['content']}\n")

                # Extract local feedback
                feedback = LocalFeedback(assistant_message["content"])
                feedbacks.append(feedback)
                print(f"Agent {i+1} Feedback:\n{feedback}\n")

            # Aggregate Feedback
            aggregated_feedback = FederatedAggregate(feedbacks)
            print(f"Aggregated Feedback after Round {t}, Sub-iteration {m}:\n{aggregated_feedback}\n")

            # Broadcast Feedback to All Agents
            broadcast_message = BroadcastFeedback(aggregated_feedback)
            for i in range(k):
                agent_contexts[i].append(broadcast_message)
        print("\n")

    print(f"=== Reasoned Phase: Round {N} ===\n")
    for m in range(1, M+1):
        print(f"-- Sub-iteration {m} --")
        feedbacks = []
        # Construct the input for this sub-iteration
        for i in range(k):
            # Construct message with reasoned prompt and aggregated feedback
            user_message = construct_message(P_R, aggregated_feedback)
            agent_contexts[i].append(user_message)

            # Generate reasoned response
            completion = generate_answer(agent_contexts[i])
            assistant_message = construct_assistant_message(completion)
            agent_contexts[i].append(assistant_message)
            print(f"Agent {i+1} ({agent_roles[i]}) Reasoned Response:\n{assistant_message['content']}\n")

            # Extract local feedback
            feedback = LocalFeedback(assistant_message["content"])
            feedbacks.append(feedback)
            print(f"Agent {i+1} Feedback:\n{feedback}\n")

        # Aggregate Feedback
        aggregated_feedback = FederatedAggregate(feedbacks)
        print(f"Aggregated Feedback after Reasoned Round {N}, Sub-iteration {m}:\n{aggregated_feedback}\n")

        # Broadcast Feedback to All Agents
        broadcast_message = BroadcastFeedback(aggregated_feedback)
        for i in range(k):
            agent_contexts[i].append(broadcast_message)
    print("\n")

    print("=== Final Response Phase ===\n")
    final_answer = agent_contexts[k-1][-2]["content"]  

    print(f"Final Response:\n{final_answer}\n")
    # -------------------------
    # Serialize or Save Conversation Data
    # -------------------------
    generated_description = agent_contexts
    with open(f"agents{k}_rounds{N}_sub_iters{M}.pkl", "wb") as f:
        pickle.dump(generated_description, f)

    print("Aggregation complete.")

if __name__ == "__main__":
    main()