<!--npm run lint -- --fix -->

<template>
  <b-container>
    <div id="german">
      <b-jumbotron
        bg-variant="info"
        text-variant="Secondary"
        border-variant="dark"
      >
        <template #header>Statlog (German Credit Data)</template>
        <template #lead>
          This dataset classifies people described by a set of attributes as
          good or bad credit risks.
        </template>
        <hr />
        Click on this button to select the Model to explain and insert the
        Values of the German Credit Features to predict its Class:
        <br />
        <br />
        <b-button variant="success" v-b-toggle.sidebar-1>Input Values</b-button>
        <hr />
        <b-row md="3">
          <b-col></b-col>
          <b-col class="position-relative">
            <b-card
              border-variant="secondary"
              header="The Credit Risk is: "
              bg-variant="primary"
              text-variant="white"
              align="center"
              v-if="class_german"
            >
              <b-card-text
                ><h4>
                  <b-badge variant="dark">{{
                    class_german.class_german
                  }}</b-badge>
                </h4>
                <hr />
                With the Probability of:
                <br />
                <h5>
                  <b-badge variant="dark"
                    >{{ class_german.class_prob }}%</b-badge
                  >
                </h5></b-card-text
              >
            </b-card>
          </b-col>
          <b-col></b-col>
        </b-row>
        <br />
        <b-sidebar id="sidebar-1" title="German Credit" shadow="true">
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
                id="ig-duration_in_month"
                label="Duration in month:"
                label-for="i-duration_in_month"
              >
                <b-form-input
                  id="i-duration_in_month"
                  size="sm"
                  v-model="form.duration_in_month"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-credit_amount"
                label="Credit amount:"
                label-for="i-credit_amount"
              >
                <b-form-input
                  id="i-credit_amount"
                  size="sm"
                  v-model="form.credit_amount"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-installment_as_income_perc"
                label="Installment rate in percentage of disposable income (installment_as_income_perc):"
                label-for="i-installment_as_income_perc"
              >
                <b-form-input
                  id="i-installment_as_income_perc"
                  size="sm"
                  v-model="form.installment_as_income_perc"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-present_res_since"
                label="Present residence since (present_res_since):"
                label-for="i-present_res_since"
              >
                <b-form-input
                  id="i-present_res_since"
                  size="sm"
                  v-model="form.present_res_since"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />

              <b-form-group id="ig-age" label="Age:" label-for="i-age">
                <b-form-input
                  id="i-age"
                  size="sm"
                  v-model="form.age"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-credits_this_bank"
                label="Number of existing credits at this bank (credits_this_bank):"
                label-for="i-credits_this_bank"
              >
                <b-form-input
                  id="i-credits_this_bank"
                  size="sm"
                  v-model="form.credits_this_bank"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-people_under_maintenance"
                label="Number of people being liable to provide maintenance for (people_under_maintenance):"
                label-for="i-people_under_maintenance"
              >
                <b-form-input
                  id="i-people_under_maintenance"
                  size="sm"
                  v-model="form.people_under_maintenance"
                  type="number"
                  min="0"
                  step="1"
                  required
                ></b-form-input>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-account_check_status"
                label="Status of existing checking account (account_check_status):"
                label-for="i-account_check_status"
              >
                <b-form-select
                  id="i-account_check_status"
                  v-model="form.account_check_status"
                  :options="account_check_status_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-credit_history"
                label="Credit history:"
                label-for="i-credit_history"
              >
                <b-form-select
                  id="i-credit_history"
                  v-model="form.credit_history"
                  :options="credit_history_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-purpose"
                label="Purpose:"
                label-for="i-purpose"
              >
                <b-form-select
                  id="i-purpose"
                  v-model="form.purpose"
                  :options="purpose_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-savings"
                label="Savings account/bonds (savings):"
                label-for="i-savings"
              >
                <b-form-select
                  id="i-savings"
                  v-model="form.savings"
                  :options="savings_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-present_emp_since"
                label="Present employment since (present_emp_since):"
                label-for="i-present_emp_since"
              >
                <b-form-select
                  id="i-present_emp_since"
                  v-model="form.present_emp_since"
                  :options="present_emp_since_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-personal_status_sex"
                label="Personal status and sex:"
                label-for="i-personal_status_sex"
              >
                <b-form-select
                  id="i-personal_status_sex"
                  v-model="form.personal_status_sex"
                  :options="personal_status_sex_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-other_debtors"
                label="Other debtors / guarantors (other_debtors):"
                label-for="i-other_debtors"
              >
                <b-form-select
                  id="i-other_debtors"
                  v-model="form.other_debtors"
                  :options="other_debtors_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-property"
                label="Property:"
                label-for="i-property"
              >
                <b-form-select
                  id="i-property"
                  v-model="form.property"
                  :options="property_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-other_installment_plans"
                label="Other installment plans:"
                label-for="i-other_installment_plans"
              >
                <b-form-select
                  id="i-other_installment_plans"
                  v-model="form.other_installment_plans"
                  :options="other_installment_plans_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-housing"
                label="Housing:"
                label-for="i-housing"
              >
                <b-form-select
                  id="i-housing"
                  v-model="form.housing"
                  :options="housing_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group id="ig-job" label="Job:" label-for="i-job">
                <b-form-select
                  id="i-job"
                  v-model="form.job"
                  :options="job_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-telephone"
                label="Telephone:"
                label-for="i-telephone"
              >
                <b-form-select
                  id="i-telephone"
                  v-model="form.telephone"
                  :options="telephone_options"
                  size="sm"
                  class="mt-3"
                  required
                ></b-form-select>
              </b-form-group>
              <hr />
              <br />

              <b-form-group
                id="ig-foreign_worker"
                label="Foreign worker:"
                label-for="i-foreign_worker"
              >
                <b-form-select
                  id="i-foreign_worker"
                  v-model="form.foreign_worker"
                  :options="foreign_worker_options"
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
  name: "German_credit",
  data() {
    return {
      form_chat: {
        user_input: "",
      },
      form: {
        model_to_explain: null,
        duration_in_month: 0,
        credit_amount: 0,
        installment_as_income_perc: 0,
        present_res_since: 0,
        age: 0,
        credits_this_bank: 0,
        people_under_maintenance: 0,
        account_check_status: null,
        credit_history: null,
        purpose: null,
        savings: null,
        present_emp_since: null,
        personal_status_sex: null,
        other_debtors: null,
        property: null,
        other_installment_plans: null,
        housing: null,
        job: null,
        telephone: null,
        foreign_worker: null,
      },
      account_check_status_options: [
        { value: null, text: "Please select an option" },
        { value: "0 <= ... < 200 DM", text: "0 <= ... < 200 DM" },
        { value: "< 0 DM", text: "< 0 DM" },
        {
          value: ">= 200 DM / salary assignments for at least 1 year",
          text: ">= 200 DM / salary assignments for at least 1 year",
        },
        { value: "no checking account", text: "no checking account" },
      ],
      credit_history_options: [
        { value: null, text: "Please select an option" },
        {
          value: "all credits at this bank paid back duly",
          text: "all credits at this bank paid back duly",
        },
        {
          value: "critical account/ other credits existing (not at this bank)",
          text: "critical account/ other credits existing (not at this bank)",
        },
        {
          value: "delay in paying off in the past",
          text: "delay in paying off in the past",
        },
        {
          value: "existing credits paid back duly till now",
          text: "existing credits paid back duly till now",
        },
        {
          value: "no credits taken/ all credits paid back duly",
          text: "no credits taken/ all credits paid back duly",
        },
      ],
      purpose_options: [
        { value: null, text: "Please select an option" },
        {
          value: "(vacation - does not exist?)",
          text: "(vacation - does not exist?)",
        },
        { value: "business", text: "business" },
        { value: "car (new)", text: "car (new)" },
        { value: "car (used)", text: "car (used)" },
        { value: "domestic appliances", text: "domestic appliances" },
        { value: "education", text: "education" },
        { value: "furniture/equipment", text: "furniture/equipment" },
        { value: "radio/television", text: "radio/television" },
        { value: "repairs", text: "repairs" },
        { value: "retraining", text: "retraining" },
      ],
      savings_options: [
        { value: null, text: "Please select an option" },
        { value: ".. >= 1000 DM ", text: ".. >= 1000 DM " },
        { value: "... < 100 DM", text: "... < 100 DM" },
        { value: "100 <= ... < 500 DM", text: "100 <= ... < 500 DM" },
        { value: "500 <= ... < 1000 DM ", text: "500 <= ... < 1000 DM " },
        {
          value: "unknown/ no savings account",
          text: "unknown/ no savings account",
        },
      ],
      present_emp_since_options: [
        { value: null, text: "Please select an option" },
        { value: ".. >= 7 years", text: ".. >= 7 years" },
        { value: "... < 1 year ", text: "... < 1 year " },
        { value: "1 <= ... < 4 years", text: "1 <= ... < 4 years" },
        { value: "4 <= ... < 7 years", text: "4 <= ... < 7 years" },
        { value: "unemployed", text: "unemployed" },
      ],
      personal_status_sex_options: [
        { value: null, text: "Please select an option" },
        {
          value: "female : divorced/separated/married",
          text: "female : divorced/separated/married",
        },
        {
          value: "male : divorced/separated",
          text: "male : divorced/separated",
        },
        { value: "male : married/widowed", text: "male : married/widowed" },
        { value: "male : single", text: "male : single" },
      ],
      other_debtors_options: [
        { value: null, text: "Please select an option" },
        { value: "co-applicant", text: "co-applicant" },
        { value: "guarantor", text: "guarantor" },
        { value: "none", text: "none" },
      ],
      property_options: [
        { value: null, text: "Please select an option" },
        {
          value:
            "if not A121 : building society savings agreement/ life insurance",
          text: "if not A121 : building society savings agreement/ life insurance",
        },
        {
          value: "if not A121/A122 : car or other, not in attribute 6",
          text: "if not A121/A122 : car or other, not in attribute 6",
        },
        { value: "real estate", text: "real estate" },
        { value: "unknown / no property", text: "unknown / no property" },
      ],
      other_installment_plans_options: [
        { value: null, text: "Please select an option" },
        { value: "bank", text: "bank" },
        { value: "none", text: "none" },
        { value: "stores", text: "stores" },
      ],
      housing_options: [
        { value: null, text: "Please select an option" },
        { value: "for free", text: "for free" },
        { value: "own", text: "own" },
        { value: "rent", text: "rent" },
      ],
      job_options: [
        { value: null, text: "Please select an option" },
        {
          value:
            "management/ self-employed/ highly qualified employee/ officer",
          text: "management/ self-employed/ highly qualified employee/ officer",
        },
        {
          value: "skilled employee / official",
          text: "skilled employee / official",
        },
        {
          value: "unemployed/ unskilled - non-resident",
          text: "unemployed/ unskilled - non-resident",
        },
        { value: "unskilled - resident", text: "unskilled - resident" },
      ],
      telephone_options: [
        { value: null, text: "Please select an option" },
        { value: "none", text: "none" },
        {
          value: "yes, registered under the customers name ",
          text: "yes, registered under the customers name ",
        },
      ],
      foreign_worker_options: [
        { value: null, text: "Please select an option" },
        { value: "no", text: "no" },
        { value: "yes", text: "yes" },
      ],
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
      class_german: null,
      explanation: null,
    };
  },
  methods: {
    start_to_explain_instance(event) {
      event.preventDefault();
      this.invalid_start = true;
      getSingleEndpoint(this.form, "german_lore_explanation").then((res) => {
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
      getSingleEndpoint(this.form, "german_lore").then((res) => {
        this.class_german = res.data;
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
