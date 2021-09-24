<!--npm run lint -- --fix -->

<template>
  <b-container>
    <div id="compas">
      <b-jumbotron
        bg-variant="info"
        text-variant="Secondary"
        border-variant="dark"
      >
        <template #header>COMPAS</template>
        <template #lead>
          Correctional Offender Management Profiling for Alternative Sanctions
        </template>
        <hr />
        Click on this button to select the Model to explain and insert the
        Values of the COMPAS Features to predict its Class:
        <br />
        <br />
        <b-button variant="success" v-b-toggle.sidebar-1>Input Values</b-button>
        <hr />
        <b-row md="3">
          <b-col></b-col>
          <b-col class="position-relative">
            <b-card
              border-variant="secondary"
              header="The COMPAS Risk Score is: "
              bg-variant="primary"
              text-variant="white"
              align="center"
              v-if="class_compas"
            >
              <b-card-text
                ><h4>
                  <b-badge variant="dark">{{
                    class_compas.class_compas
                  }}</b-badge>
                </h4>
                <hr />
                With the Probability of:
                <br />
                <h5>
                  <b-badge variant="dark"
                    >{{ class_compas.class_prob }}%</b-badge
                  >
                </h5></b-card-text
              >
            </b-card>
          </b-col>
          <b-col></b-col>
        </b-row>
        <br />
        <b-sidebar id="sidebar-1" title="COMPAS" shadow="true">
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
              <b-form-group id="ig-age" label="Age:" label-for="i-age">
                <b-form-input
                  id="i-age"
                  size="sm"
                  v-model="form.age"
                  type="number"
                  min="18"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-priors_count"
                label="Number of Priors Arrests: (priors_count)"
                label-for="i-priors_count"
              >
                <b-form-input
                  id="i-priors_count"
                  size="sm"
                  v-model="form.priors_count"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-days_b_screening_arrest"
                label="Days be Screening Arrest:"
                label-for="i-days_b_screening_arrest"
              >
                <b-form-input
                  id="i-days_b_screening_arrest"
                  size="sm"
                  v-model="form.days_b_screening_arrest"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-is_recid"
                label="Is Recidivist: (is_recid>0.5 => Yes, is_recid<0.5 => No)"
                label-for="i-is_recid"
              >
                <b-form-select
                  id="i-is_recid"
                  v-model="form.is_recid"
                  :options="is_recid_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-is_violent_recid"
                label="Is Violent Recidivist: (is_violent_recid>0.5 => Yes, is_violent_recid<0.5 => No)"
                label-for="i-is_violent_recid"
              >
                <b-form-select
                  id="i-is_violent_recid"
                  v-model="form.is_violent_recid"
                  :options="is_violent_recid_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-two_year_recid"
                label="Has Recidivated in Two Years: (two_year_recid>0.5 => Yes, two_year_recid<0.5 => No)"
                label-for="i-two_year_recid"
              >
                <b-form-select
                  id="i-two_year_recid"
                  v-model="form.two_year_recid"
                  :options="two_year_recid_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-length_of_stay"
                label="Length of Stay in Days:"
                label-for="i-length_of_stay"
              >
                <b-form-input
                  id="i-length_of_stay"
                  size="sm"
                  v-model="form.length_of_stay"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-age_cat"
                label="Age Category:"
                label-for="i-age_cat"
              >
                <b-form-select
                  id="i-age_cat"
                  v-model="form.age_cat"
                  :options="age_cat_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />
              <b-form-group id="ig-sex" label="Sex:" label-for="i-sex">
                <b-form-select
                  id="i-sex"
                  v-model="form.sex"
                  :options="sex_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />
              <b-form-group id="ig-race" label="Race:" label-for="i-race">
                <b-form-select
                  id="i-race"
                  v-model="form.race"
                  :options="race_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />
              <b-form-group
                id="ig-c_charge_degree"
                label="Felony or Misdemeanor Charge (Charge Degree):"
                label-for="i-c_charge_degree"
              >
                <b-form-select
                  id="i-c_charge_degree"
                  v-model="form.c_charge_degree"
                  :options="c_charge_degree_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <br />
              <hr />
              <b-button type="submit" variant="primary" :disabled="invalid"
                >Predict</b-button
              >
<!--              <b-overlay no-wrap :show="invalid"></b-overlay>-->
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
<!--          <b-overlay no-wrap :show="invalid_start"></b-overlay>-->
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
<!--              <b-overlay no-wrap :show="invalid_chat"></b-overlay>-->
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
  name: "Compas",
  data() {
    return {
      form_chat: {
        user_input: "",
      },
      form: {
        model_to_explain: null,
        age: 18,
        priors_count: 0,
        days_b_screening_arrest: 0,
        is_recid: null,
        is_violent_recid: null,
        two_year_recid: null,
        length_of_stay: 0,
        age_cat: null,
        sex: null,
        race: null,
        c_charge_degree: null,
      },
      is_recid_options: [
        { value: null, text: "Please select an option" },
        { value: 0, text: "No" },
        { value: 1, text: "Yes" },
      ],
      is_violent_recid_options: [
        { value: null, text: "Please select an option" },
        { value: 0, text: "No" },
        { value: 1, text: "Yes" },
      ],
      two_year_recid_options: [
        { value: null, text: "Please select an option" },
        { value: 0, text: "No" },
        { value: 1, text: "Yes" },
      ],
      age_cat_options: [
        { value: null, text: "Please select an option" },
        { value: "Less than 25", text: "Less than 25" },
        { value: "25 - 45", text: "25 - 45" },
        { value: "Greater than 45", text: "Greater than 45" },
      ],
      sex_options: [
        { value: null, text: "Please select an option" },
        { value: "Male", text: "Male" },
        { value: "Female", text: "Female" },
      ],
      race_options: [
        { value: null, text: "Please select an option" },
        { value: "African-American", text: "African-American" },
        { value: "Asian", text: "Asian" },
        { value: "Caucasian", text: "Caucasian" },
        { value: "Hispanic", text: "Hispanic" },
        { value: "Native American", text: "Native American" },
        { value: "Other", text: "Other" },
      ],
      c_charge_degree_options: [
        { value: null, text: "Please select an option" },
        { value: "F", text: "F" },
        { value: "M", text: "M" },
      ],
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
      can_explain: false,
      can_xbot_respond: false,
      invalid_start: false,
      invalid: false,
      invalid_chat: false,
      can_chat: false,
      class_compas: null,
      explanation: null,
    };
  },
  methods: {
    start_to_explain_instance(event) {
      event.preventDefault();
      this.invalid_start = true;
      getSingleEndpoint(this.form, "compas_lore_explanation").then((res) => {
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
      getSingleEndpoint(this.form, "compas_lore").then((res) => {
        this.class_compas = res.data;
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
