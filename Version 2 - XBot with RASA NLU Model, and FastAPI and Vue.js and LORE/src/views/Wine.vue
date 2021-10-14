<!--npm run lint -- --fix -->

<template>
  <b-container>
    <div id="wine">
      <b-jumbotron
        bg-variant="info"
        text-variant="Secondary"
        border-variant="dark"
      >
        <template #header>Wine</template>
        <template #lead> The Wine Quality Dataset </template>
        <hr />
        Click on this button to select the Model to explain and insert the
        Values of the Wine Features to predict its Class:
        <br />
        <br />
        <b-button variant="success" v-b-toggle.sidebar-1>Input Values</b-button>
        <hr />
        <b-row md="3">
          <b-col></b-col>
          <b-col class="position-relative">
            <b-card
              border-variant="secondary"
              header="The Wine Quality Class is: "
              bg-variant="primary"
              text-variant="white"
              align="center"
              v-if="class_wine"
            >
              <b-card-text
                ><h4>
                  <b-badge variant="dark">{{ class_wine.class_wine }}</b-badge>
                </h4>
                <hr />
                With the Probability of:
                <br />
                <h5>
                  <b-badge variant="dark">{{ class_wine.class_prob }}%</b-badge>
                </h5></b-card-text
              >
            </b-card>
          </b-col>
          <b-col></b-col>
        </b-row>
        <br />
        <b-sidebar id="sidebar-1" title="wine" shadow="true">
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
                id="ig-Alcohol"
                label="Alcohol:"
                label-for="i-Alcohol"
              >
                <b-form-input
                  id="i-Alcohol"
                  size="sm"
                  v-model="form.Alcohol"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-Malic_acid"
                label="Malic acid:"
                label-for="i-Malic_acid"
              >
                <b-form-input
                  id="i-Malic_acid"
                  size="sm"
                  v-model="form.Malic_acid"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group id="ig-Ash" label="Ash:" label-for="i-Ash">
                <b-form-input
                  id="i-Ash"
                  size="sm"
                  v-model="form.Ash"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-Acl"
                label="Acl (Alcalinity of ash):"
                label-for="i-Acl"
              >
                <b-form-input
                  id="i-Acl"
                  size="sm"
                  v-model="form.Acl"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group id="ig-Mg" label="Mg (Magnesium):" label-for="i-Mg">
                <b-form-input
                  id="i-Mg"
                  size="sm"
                  v-model="form.Mg"
                  type="number"
                  min="0.0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-Phenols"
                label="Phenols (Total phenols):"
                label-for="i-Phenols"
              >
                <b-form-input
                  id="i-Phenols"
                  size="sm"
                  v-model="form.Phenols"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-Flavanoids"
                label="Flavanoids:"
                label-for="i-Flavanoids"
              >
                <b-form-input
                  id="i-Flavanoids"
                  size="sm"
                  v-model="form.Flavanoids"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-Nonflavanoid_phenols"
                label="Nonflavanoid phenols:"
                label-for="i-Nonflavanoid_phenols"
              >
                <b-form-input
                  id="i-Nonflavanoid_phenols"
                  size="sm"
                  v-model="form.Nonflavanoid_phenols"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-Proanth"
                label="Proanthocyanins (Proanth):"
                label-for="i-Proanth"
              >
                <b-form-input
                  id="i-Proanth"
                  size="sm"
                  v-model="form.Proanth"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-Color_int"
                label="Color intensity (Color_int):"
                label-for="i-Color_int"
              >
                <b-form-input
                  id="i-Color_int"
                  size="sm"
                  v-model="form.Color_int"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group id="ig-Hue" label="Hue:" label-for="i-Hue">
                <b-form-input
                  id="i-Hue"
                  size="sm"
                  v-model="form.Hue"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-OD"
                label="OD280/OD315 of diluted wines (OD):"
                label-for="i-OD"
              >
                <b-form-input
                  id="i-OD"
                  size="sm"
                  v-model="form.OD"
                  type="number"
                  min="0.0"
                  step="0.01"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-Proline"
                label="Proline:"
                label-for="i-Proline"
              >
                <b-form-input
                  id="i-Proline"
                  size="sm"
                  v-model="form.Proline"
                  type="number"
                  min="0.0"
                  step="1"
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
          <b-card
            v-show="can_explain"
            bg-variant="dark"
            text-variant="white"
            title="X-Bot"
          >
            <b-card-text>
              Click on this button to explain the selected instance and start to
              chat with X-Bot:
            </b-card-text>
            <b-button v-on:click="start_to_explain_instance" variant="success"
              >Start</b-button
            >
          </b-card>
          <b-overlay no-wrap :show="invalid_start"></b-overlay>
        </b-col>
      </b-row>
      <b-card bg-variant="light" border-variant="dark" v-show="can_chat">
        <b-row md="2">
          <b-col>
            <b-form @submit="onSubmit_chat">
              <b-form-group
                id="ig-user_input"
                label="Chat with X-Bot:"
                label-for="i-user_input"
              >
                <b-form-input
                  id="i-user_input"
                  size="lg"
                  placeholder="Ask me your question"
                  v-model="form_chat.user_input"
                  type="text"
                  required
                ></b-form-input>
              </b-form-group>
              <b-button type="submit" variant="primary" :disabled="invalid_chat"
                >Send</b-button
              >
              <b-overlay no-wrap :show="invalid_chat"></b-overlay>
            </b-form>
          </b-col>
          <b-col>
            <b-card
              border-variant="secondary"
              header="X-Bot: "
              bg-variant="primary"
              text-variant="white"
              align="center"
              v-if="explanation"
              v-show="can_xbot_respond"
            >
              <b-card-text>{{ explanation.xbot_explanation }}</b-card-text>
              <hr />
              <b-card-text
                ><strong>Intent:</strong>
                {{ explanation.tag_intent }}</b-card-text
              >
            </b-card>
          </b-col>
        </b-row>
      </b-card>
    </div>
    <br />
  </b-container>
</template>

<script>
import { getSingleEndpoint } from "@/axiosInstance";

export default {
  name: "Wine",
  data() {
    return {
      form_chat: {
        user_input: "",
      },
      form: {
        model_to_explain: null,
        Alcohol: 0,
        Malic_acid: 0,
        Ash: 0,
        Acl: 0,
        Mg: 0,
        Phenols: 0,
        Flavanoids: 0,
        Nonflavanoid_phenols: 0,
        Proanth: 0,
        Color_int: 0,
        Hue: 0,
        OD: 0,
        Proline: 0,
      },
      model_to_explain_options: [
        {
          value: "GradientBoostingClassifier",
          text: "Gradient Boosting Classifier",
        },
        { value: "RandomForestClassifier", text: "Random Forest Classifier" },
        // {
        //   value: "SGDClassifier",
        //   text: "Stochastic Gradient Descent Classifier",
        // },
        { value: "SVC", text: "Support Vector Machine Classifier" },
      ],
      can_explain: false,
      can_xbot_respond: false,
      invalid_start: false,
      invalid: false,
      invalid_chat: false,
      can_chat: false,
      explanation: null,
      class_wine: null,
    };
  },
  methods: {
    start_to_explain_instance(event) {
      event.preventDefault();
      this.invalid_start = true;
      getSingleEndpoint(this.form, "wine_lore_explanation").then((res) => {
        this.result = res.data;
        this.invalid_start = false;
        this.can_explain = false;
        this.can_chat = true;
      });
    },
    onSubmit_chat(event) {
      event.preventDefault();
      this.invalid_chat = true;
      getSingleEndpoint(this.form_chat, "chat_bot").then((res) => {
        this.explanation = res.data;
        this.invalid_chat = false;
        this.can_xbot_respond = true;
      });
    },
    onSubmit(event) {
      event.preventDefault();
      this.invalid = true;
      getSingleEndpoint(this.form, "wine_lore").then((res) => {
        this.class_wine = res.data;
        this.invalid = false;
        this.can_explain = true;
        this.can_chat = false;
        this.can_xbot_respond = false;
      });
    },
  },
};
</script>

<style scoped></style>
