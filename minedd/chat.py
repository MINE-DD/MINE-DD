from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

document_chunks = [
    "(89%) showed an inverse correlation with at least one climatological variable (18 studies, or 69% statistically significant), compared with 10 (39%) showing a positive correlation (six studies, or 23% statistically significant) with at least one climatological variable (Table 1). According to the pooled GEE analysis, low values of all climatological variables predicted increased monthly incidence of rotavirus disease in patients with gastroenteritis.",
    "According to the pooled GEE analysis, low values of all climatological variables predicted increased monthly incidence of rotavirus disease in patients with gastroenteritis. Using data on the total number of monthly rotavirus cases, rather than proportion of diarrhoea patients testing positive for rotavirus, avoided the potential for reporting on patterns driven by seasonal changes in other diarrhoea pathogens. However, we should note that similar results were found with both approaches (total case count and proportion of diarrhoea cases testing positive for rotavirus).",
    "diarrhoea pathogens. However, we should note that similar results were found with both approaches (total case count and proportion of diarrhoea cases testing positive for rotavirus). The effect of seasonal changes on rotavirus incidence seen here is not as extreme in the tropics as it is in temperate areas of the world. Rotavirus is found year-round in the tropics with peaks and valleys, whereas incidence often goes to zero in some months in temperate areas. One explanation for this phenomenon is that less climatic variability exists in tropical climates and zones, so variations in climatological variables are not large enough to cause the observed effect. Still, the",
    "climatic variability exists in tropical climates and zones, so variations in climatological variables are not large enough to cause the observed effect. Still, the fact that rotavirus persists year-round in tropical areas of the world, and that rotavirus responds to climatic changes in many different climatic zones throughout the world, suggests that it is not an absolute temperature or humidity level that favors rotavirus transmission, but rather a relative change in climatic conditions. We see a large amount of hete",
    "in many different climatic zones throughout the world, suggests that it is not an absolute temperature or humidity level that favors rotavirus transmission, but rather a relative change in climatic conditions. We see a large amount of heterogeneity both within and between studies in the pooled analysis. The significant unexplained variation is a limitation of the study. The heterogeneity suggests that we would expect to see a stronger effect, and therefore have greater predictive power, if we could reduce some of the sources of variation between the different studies reviewed, such as socioeconomic status of patients, age of patients, sampling scheme, diagnostic methods used,",
    "if we could reduce some of the sources of variation between the different studies reviewed, such as socioeconomic status of patients, age of patients, sampling scheme, diagnostic methods used, lengths of studies, numbers of participants sampled, populations of study regions and differing climatic conditions at each study location. Most studies included only children while others included patients",
    "of all ages. Also, the majority of studies reviewed were carried out for 2 years or less, which is a relatively short period of time to capture the effects of seasonality; studies of longer duration are preferable for establishing the relationship between climate and rotavirus disease. While all studies lie within the latitudes defined as the tropics, various climatic regimes (e.g. rainforest vs semi-arid)",
    "ing the relationship between climate and rotavirus disease. While all studies lie within the latitudes defined as the tropics, various climatic regimes (e.g. rainforest vs semi-arid) prevail in the different settings and at different altitudes, potentially confounding the results. We were unable to account for these differences in our analysis of potential covariates. Understanding rotavirus transmission The heterogeneity in effect observed in the pooled analysis is not surprising given that this analysis did not take into account additional factors potentially",
    "analysis is not surprising given that this analysis did not take into account additional factors potentially affecting rotavirus transmission, such as sanitation and hygiene practices or flood peaks. Several authors of the articles reviewed noted multiple peaks in rotavirus incidence as affected by the monsoon rains (Table 1). Flooding in conjunction with poor sanitation could augment the waterborne component of rotavirus transmission, obfuscating",
    "Flooding in conjunction with poor sanitation could augment the waterborne component of rotavirus transmission, obfuscating the seasonal patterns, which might be driven more by other routes of transmission, such as the air or fomites. Strong evidence suggests that rotavirus is a waterborne pathogen. The virus can retain its infectivity for several days in aqueous environments, and waterborne spread has been implicated in a number of rotavirus outbrea",
    "fomites. Strong evidence suggests that rotavirus is a waterborne pathogen. The virus can retain its infectivity for several days in aqueous environments, and waterborne spread has been implicated in a number of rotavirus outbreaks.7 However, the high rates of infection in the first 3 years of life regardless of sanitary conditions, the failure to document fecal-oral transmission in several outbreaks of rotavirus diarrhoea, and the dramatic spread of rotavirus over large geographic areas in the winter",
    "failure to document fecal-oral transmission in several outbreaks of rotavirus diarrhoea, and the dramatic spread of rotavirus over large geographic areas in the winter in temperate zones suggests that water alone may not be responsible for all rotavirus transmission.5 No direct evidence shows that fomites and environmental surfaces play a role in the spread of rotavirus gastroe",
    "No direct evidence shows that fomites and environmental surfaces play a role in the spread of rotavirus gastroenteritis, but indirect evidence shows that these possess a strong potential for spreading rotavirus gastroenteritis. Rotaviruses can remain viable on inanimate surfaces for several days when dried from a fecal suspension.7 Many authors have also suggested that rotavirus spreads through the air. In nosocomial outbreaks of rotavirus gastroenteritis, many patients show"
]

chunks_listed = "\n\n".join(f"Chunk {i + 1}) {chunk}" for i, chunk in enumerate(document_chunks))


class Chat:
    def __init__(self, model_name="llama3.2:latest"):
        self.model_name = model_name
        self.model = ChatOllama(model=model_name, temperature=0.0, max_tokens=512)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant. Your job is to answer the user's questions based on the provided document chunks and conversation history.
            A list fo document chunks is also provided. Your response should only be based on the conversation history and the document chunks. If you don't know the answer, say "I don't know".
            Do NOT make up an answer. Only include information from the conversation history and document chunks. Aditionally, you should mention which chunk number (or numbers) contain the information source of your answer". 
            Keep the answer to the question as factual and concise as possible.

            Document Chunks:
            {chunks_listed}

            Conversation History:
            {context}

            Question:
            {question}

            Answer:
            """
        )
        self.chain = self.prompt | self.model
        self.history = []

    def ask(self, question):
        context = "\n".join(
            f"User: {q}\nAI: {a}" for q, a in self.history
        ) if self.history else "The conversation has just begun."
        response = self.chain.invoke({"context": context, "question": question, "chunks_listed": chunks_listed})
        self.history.append((question, response.text().strip()))
        if len(self.history) > 10:
            self.history.pop(0)
        return response.text().strip()

    def chat_loop(self):
        print("\n--- Mine-DD: Chat-with-Docs ---")
        print("Type 'exit' to quit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print(f"{self.model_name}: Goodbye!")
                break
            print(f"{self.model_name}: Thinking...\n")
            try:
                answer = self.ask(user_input)
                print(f"{self.model_name}: {answer}\n")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    chat = Chat()
    chat.chat_loop()
