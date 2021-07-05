<!--npm run lint -- --fix -->

<template>
  <b-container>
    <div id="iris">
      <b-jumbotron
        bg-variant="info"
        text-variant="Secondary"
        border-variant="dark"
      >
        <template #header>Iris</template>
        <template #lead> The Iris Flower Dataset </template>
        <hr />
        Click on this button to select the Model to explain and insert the
        Values of the Iris Features to predict its Class:
        <br />
        <br />
        <b-button variant="success" v-b-toggle.sidebar-1>Input Values</b-button>
        <hr />
        <b-row md="2">
          <b-col class="position-relative">
            <b-card
              border-variant="secondary"
              header="Iris Class: "
              header-border-variant="secondary"
              align="center"
              v-if="class_iris"
            >
              <b-card-text
                ><h4>
                  <b-badge variant="dark">{{ class_iris.class_iris }}</b-badge>
                </h4>
                <br />
                With the Probability of:
                <br />
                <h5>
                  <b-badge variant="dark">{{ class_iris.class_prob }}</b-badge>
                </h5>
              </b-card-text>
            </b-card>
          </b-col>
          <b-col>
            <b-card
              border-variant="secondary"
              header="X-Bot: "
              header-border-variant="secondary"
              align="center"
              v-if="explanation"
            >
              <b-card-text>{{ explanation.xbot_explanation }}</b-card-text>
              <b-card-text
                ><strong>Intent:</strong>
                {{ explanation.tag_intent }}</b-card-text
              >
            </b-card>
          </b-col>
        </b-row>
        <br />
        <b-sidebar id="sidebar-1" title="Iris" backdrop shadow="true">
          <div class="px-3 py-2 text-left">
            <b-form @submit="onSubmit">
              <b-form-group
                id="ig-explainer_model"
                label="Model:"
                label-for="ig-explainer_model"
              >
                <b-form-radio-group
                  id="ig-explainer_model"
                  v-model="form.model_to_explain"
                  :options="model_to_explain_options"
                  required
                ></b-form-radio-group>
              </b-form-group>
              <hr />
              <hr />
              <br />
              <b-form-group
                id="ig-sepal_length"
                label="Sepal Length:"
                label-for="i-sepal_length"
              >
                <b-form-input
                  id="i-sepal_length"
                  size="sm"
                  v-model="form.sepal_length"
                  type="number"
                  min="0.1"
                  step="0.1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-sepal_width"
                label="Sepal Width:"
                label-for="i-sepal_width"
              >
                <b-form-input
                  id="i-sepal_width"
                  size="sm"
                  v-model="form.sepal_width"
                  type="number"
                  min="0.1"
                  step="0.1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-petal_length"
                label="Petal Length:"
                label-for="i-petal_length"
              >
                <b-form-input
                  id="i-petal_length"
                  size="sm"
                  v-model="form.petal_length"
                  type="number"
                  min="0.1"
                  step="0.1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-petal_width"
                label="Petal Width:"
                label-for="i-petal_width"
              >
                <b-form-input
                  id="i-petal_width"
                  size="sm"
                  v-model="form.petal_width"
                  type="number"
                  min="0.1"
                  step="0.1"
                  required
                ></b-form-input>
              </b-form-group>
              <br />
              <hr />
              <b-button type="submit" variant="primary" :disabled="invalid"
                >Predict</b-button
              >
              <b-overlay no-wrap :show="invalid"></b-overlay>
            </b-form>
          </div>
        </b-sidebar>
      </b-jumbotron>
      <b-row>
        <b-col>
          <b-form @submit="onSubmit_chat">
            <b-form-group
              id="ig-user_input"
              label="Chat with X-Bot:"
              label-for="i-user_input"
              v-show="can_chat"
            >
              <b-form-input
                id="i-user_input"
                size="lg"
                placeholder="Ask me your question"
                v-model="form_chat.user_input"
                type="text"
                required
                v-show="can_chat"
              ></b-form-input>
            </b-form-group>
            <b-button
              type="submit"
              variant="primary"
              :disabled="invalid_chat"
              v-show="can_chat"
              >Send</b-button
            >
            <b-overlay no-wrap :show="invalid_chat"></b-overlay>
          </b-form>
        </b-col>
      </b-row>
    </div>
  </b-container>
</template>

<script>
import { getSingleEndpoint } from "@/axiosInstance";

export default {
  name: "Iris",
  data() {
    return {
      form_chat: {
        user_input: "",
      },
      form: {
        model_to_explain: null,
        sepal_length: 0,
        sepal_width: 0,
        petal_length: 0,
        petal_width: 0,
      },
      model_to_explain_options: [
        {
          value: "GradientBoostingClassifier",
          text: "Gradient Boosting Classifier",
        },
        { value: "RandomForestClassifier", text: "Random Forest Classifier" },
        {
          value: "SGDClassifier",
          text: "Stochastic Gradient Descent Classifier",
        },
        { value: "SVC", text: "Support Vector Machine Classifier" },
      ],
      invalid: false,
      invalid_chat: false,
      can_chat: false,
      explanation: null,
      class_iris: null,
    };
  },
  methods: {
    onSubmit_chat(event) {
      event.preventDefault();
      this.invalid_chat = true;
      getSingleEndpoint(this.form_chat, "chat_bot").then((res) => {
        this.explanation = res.data;
        this.invalid_chat = false;
      });
    },
    onSubmit(event) {
      event.preventDefault();
      this.invalid = true;
      getSingleEndpoint(this.form, "iris_lore").then((res) => {
        this.class_iris = res.data;
        this.invalid = false;
        this.can_chat = true;
      });
    },
  },
};
</script>

<style scoped></style>
