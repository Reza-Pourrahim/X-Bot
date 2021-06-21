<!--npm run lint -- --fix -->

<template>
  <b-container>
    <div id="adult">
      <b-jumbotron
        bg-variant="info"
        text-variant="Secondary"
        border-variant="dark"
      >
        <template #header>Adult Income</template>
        <template #lead>
          Predict whether income exceeds $50K/yr based on census data. Also
          known as "Census Income" dataset.
        </template>
        <hr />
        <b-button v-b-toggle.sidebar-1>Input Values</b-button>
        <hr />
        <b-row md="2">
          <b-col class="position-relative">
            <b-card
              border-variant="secondary"
              header="Income Class: "
              header-border-variant="secondary"
              align="center"
              v-if="class_adult"
            >
              <b-card-text
                ><h4>
                  <b-badge variant="dark">{{
                    class_adult.class_adult
                  }}</b-badge>
                </h4></b-card-text
              >
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
        <b-sidebar id="sidebar-1" title="Adult" backdrop shadow="true">
          <div class="px-3 py-2">
            <b-form @submit="onSubmit">
              <b-form-group id="ig-age" label="Age:" label-for="i-age">
                <b-form-input
                  id="i-age"
                  size="sm"
                  v-model="form.age"
                  type="number"
                  min="17"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>

              <b-form-group
                id="ig-capital_gain"
                label="Capital-gain:"
                label-for="i-capital_gain"
              >
                <b-form-input
                  id="i-capital_gain"
                  size="sm"
                  v-model="form.capital_gain"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>

              <b-form-group
                id="ig-capital_loss"
                label="Capital-Loss:"
                label-for="i-capital_loss"
              >
                <b-form-input
                  id="i-capital_loss"
                  size="sm"
                  v-model="form.capital_loss"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>

              <b-form-group
                id="ig-hours_per_week"
                label="Hours-per-week:"
                label-for="i-hours_per_week"
              >
                <b-form-input
                  id="i-hours_per_week"
                  size="sm"
                  v-model="form.hours_per_week"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>

              <b-form-group
                id="ig-workclass"
                label="Workclass:"
                label-for="i-workclass"
              >
                <b-form-select
                  id="i-workclass"
                  v-model="form.workclass"
                  :options="workclass_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>

              <b-form-group
                id="ig-education"
                label="Education:"
                label-for="i-education"
              >
                <b-form-select
                  id="i-education"
                  v-model="form.education"
                  :options="education_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>

              <b-form-group
                id="ig-marital_status"
                label="Marital-status:"
                label-for="i-marital_status"
              >
                <b-form-select
                  id="i-marital_status"
                  v-model="form.marital_status"
                  :options="marital_status_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>

              <b-form-group
                id="ig-occupation"
                label="Occupation:"
                label-for="i-occupation"
              >
                <b-form-select
                  id="i-occupation"
                  v-model="form.occupation"
                  :options="occupation_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>

              <b-form-group
                id="ig-relationship"
                label="Relationship:"
                label-for="i-relationship"
              >
                <b-form-select
                  id="i-relationship"
                  v-model="form.relationship"
                  :options="relationship_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>

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

              <b-form-group
                id="ig-native_country"
                label="Native_country:"
                label-for="i-native_country"
              >
                <b-form-select
                  id="i-native_country"
                  v-model="form.native_country"
                  :options="native_country_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>

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
  name: "adult",
  data() {
    return {
      form_chat: {
        user_input: "",
      },
      form: {
        age: 17,
        capital_gain: 0,
        capital_loss: 0,
        hours_per_week: 0,
        workclass: null,
        education: null,
        marital_status: null,
        occupation: null,
        relationship: null,
        race: null,
        sex: null,
        native_country: null,
      },
      workclass_options: [
        { value: null, text: "Please select an option" },
        { value: "Federal-gov", text: "Federal-gov" },
        { value: "Local-gov", text: "Local-gov" },
        { value: "Never-worked", text: "Never-worked" },
        { value: "Private", text: "Private" },
        { value: "Self-emp-inc", text: "Self-emp-inc" },
        { value: "Self-emp-not-inc", text: "Self-emp-not-inc" },
        { value: "State-gov", text: "State-gov" },
        { value: "Without-pay", text: "Without-pay" },
      ],
      education_options: [
        { value: null, text: "Please select an option" },
        { value: "10th", text: "10th" },
        { value: "11th", text: "11th" },
        { value: "12th", text: "12th" },
        { value: "1st-4th", text: "1st-4th" },
        { value: "5th-6th", text: "5th-6th" },
        { value: "7th-8th", text: "7th-8th" },
        { value: "9th", text: "9th" },
        { value: "Assoc-acdm", text: "Assoc-acdm" },
        { value: "Assoc-voc", text: "Assoc-voc" },
        { value: "Bachelors", text: "Bachelors" },
        { value: "Doctorate", text: "Doctorate" },
        { value: "HS-grad", text: "HS-grad" },
        { value: "Masters", text: "Masters" },
        { value: "Preschool", text: "Preschool" },
        { value: "Prof-school", text: "Prof-school" },
        { value: "Some-college", text: "Some-college" },
      ],
      marital_status_options: [
        { value: null, text: "Please select an option" },
        { value: "Divorced", text: "Divorced" },
        { value: "Married-AF-spouse", text: "Married-AF-spouse" },
        { value: "Married-civ-spouse", text: "Married-civ-spouse" },
        { value: "Married-spouse-absent", text: "Married-spouse-absent" },
        { value: "Never-married", text: "Never-married" },
        { value: "Separated", text: "Separated" },
        { value: "Widowed", text: "Widowed" },
      ],
      occupation_options: [
        { value: null, text: "Please select an option" },
        { value: "Adm-clerical", text: "Adm-clerical" },
        { value: "Armed-Forces", text: "Armed-Forces" },
        { value: "Craft-repair", text: "Craft-repair" },
        { value: "Exec-managerial", text: "Exec-managerial" },
        { value: "Farming-fishing", text: "Farming-fishing" },
        { value: "Handlers-cleaners", text: "Handlers-cleaners" },
        { value: "Machine-op-inspct", text: "Machine-op-inspct" },
        { value: "Other-service", text: "Other-service" },
        { value: "Priv-house-serv", text: "Priv-house-serv" },
        { value: "Prof-specialty", text: "Prof-specialty" },
        { value: "Protective-serv", text: "Protective-serv" },
        { value: "Sales", text: "Sales" },
        { value: "Tech-support", text: "Tech-support" },
        { value: "Transport-moving", text: "Transport-moving" },
      ],
      relationship_options: [
        { value: null, text: "Please select an option" },
        { value: "Husband", text: "Husband" },
        { value: "Not-in-family", text: "Not-in-family" },
        { value: "Other-relative", text: "Other-relative" },
        { value: "Own-child", text: "Own-child" },
        { value: "Unmarried", text: "Unmarried" },
        { value: "Wife", text: "Wife" },
      ],
      race_options: [
        { value: null, text: "Please select an option" },
        { value: "Amer-Indian-Eskimo", text: "Amer-Indian-Eskimo" },
        { value: "Asian-Pac-Islander", text: "Asian-Pac-Islander" },
        { value: "Black", text: "Black" },
        { value: "Other", text: "Other" },
        { value: "White", text: "White" },
      ],
      sex_options: [
        { value: null, text: "Please select an option" },
        { value: "Female", text: "Female" },
        { value: "Male", text: "Male" },
      ],
      native_country_options: [
        { value: null, text: "Please select an option" },
        { value: "Cambodia", text: "Cambodia" },
        { value: "Canada", text: "Canada" },
        { value: "China", text: "China" },
        { value: "Columbia", text: "Columbia" },
        { value: "Cuba", text: "Cuba" },
        { value: "Dominican-Republic", text: "Dominican-Republic" },
        { value: "Ecuador", text: "Ecuador" },
        { value: "El-Salvador", text: "El-Salvador" },
        { value: "England", text: "England" },
        { value: "France", text: "France" },
        { value: "Germany", text: "Germany" },
        { value: "Greece", text: "Greece" },
        { value: "Guatemala", text: "Guatemala" },
        { value: "Haiti", text: "Haiti" },
        { value: "Holand-Netherlands", text: "Holand-Netherlands" },
        { value: "Honduras", text: "Honduras" },
        { value: "Hong", text: "Hong" },
        { value: "Hungary", text: "Hungary" },
        { value: "India", text: "India" },
        { value: "Iran", text: "Iran" },
        { value: "Ireland", text: "Ireland" },
        { value: "Italy", text: "Italy" },
        { value: "Jamaica", text: "Jamaica" },
        { value: "Japan", text: "Japan" },
        { value: "Laos", text: "Laos" },
        { value: "Mexico", text: "Mexico" },
        { value: "Nicaragua", text: "Nicaragua" },
        {
          value: "Outlying-US(Guam-USVI-etc)",
          text: "Outlying-US(Guam-USVI-etc)",
        },
        { value: "Peru", text: "Peru" },
        { value: "Philippines", text: "Philippines" },
        { value: "Poland", text: "Poland" },
        { value: "Portugal", text: "Portugal" },
        { value: "Puerto-Rico", text: "Puerto-Rico" },
        { value: "Scotland", text: "Scotland" },
        { value: "South", text: "South" },
        { value: "Taiwan", text: "Taiwan" },
        { value: "Thailand", text: "Thailand" },
        { value: "Trinadad&Tobago", text: "Trinadad&Tobago" },
        { value: "United-States", text: "United-States" },
        { value: "Vietnam", text: "Vietnam" },
        { value: "Yugoslavia", text: "Yugoslavia" },
      ],
      invalid: false,
      invalid_chat: false,
      can_chat: false,
      class_adult: null,
      explanation: null,
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
      getSingleEndpoint(this.form, "adult_lore").then((res) => {
        this.class_adult = res.data;
        this.invalid = false;
        this.can_chat = true;
      });
    },
  },
};
</script>

<style scoped></style>
