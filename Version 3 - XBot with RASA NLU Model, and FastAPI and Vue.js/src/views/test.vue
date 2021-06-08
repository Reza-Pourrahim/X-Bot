<template>
  <b-container>
    <b-row>
      <b-col>
        <b-jumbotron>
          <template #header>Cardiac Risk Evaluation</template>

          <template #lead>
            This is a web interface to the GRACE classifier, to estimate Cardiac
            Risk on the basis of medical paramters. Each classification is
            provided with a rule-based explanation to highlight which paramters
            are relevant for the classification, and a set of counter rules to
            explore how to obtain a different result modifying a few key
            paramters.
            <hr />
            This is a joint work of:
            <b-avatar></b-avatar>
            <b-avatar></b-avatar>
          </template>
        </b-jumbotron>
      </b-col>
    </b-row>
    <b-row>
      <b-col>
        <h2>Risk estimator</h2>
        <p>
          Insert the parameters of the patient to estimate the cardivascular
          risk
        </p>
      </b-col>
    </b-row>
    <b-row>
      <b-col md="4">
        <b-form @submit="onSubmit">
          <b-form-group id="ig-age" label="Age" label-for="i-age">
            <b-form-input
              id="i-age"
              v-model="form.age"
              type="number"
              required
            ></b-form-input>
          </b-form-group>
          <b-form-group id="ig-hr" label="Hearth Rate" label-for="i-hr">
            <b-form-input
              id="i-hr"
              v-model="form.hr"
              type="number"
              required
            ></b-form-input>
          </b-form-group>
          <b-form-group id="ig-sbp" label="SBP" label-for="i-sbp">
            <b-form-input
              id="i-sbp"
              v-model="form.sbp"
              type="number"
              required
            ></b-form-input>
          </b-form-group>
          <b-form-group id="ig-creat" label="Creatinine" label-for="i-creat">
            <b-form-input
              id="i-creat"
              v-model="form.creat"
              type="number"
              required
            ></b-form-input>
          </b-form-group>
          <b-form-group id="ig-killip" label="KILLIP" label-for="i-killip">
            <b-form-radio-group
              id="i-killip"
              v-model="form.killip"
              :options="killipOptions"
              buttons
            ></b-form-radio-group>
          </b-form-group>
          <b-form-group id="ig-tn" label="TN" label-for="i-tn">
            <b-form-radio-group
              id="i-tn"
              v-model="form.tn"
              :options="booleanOptions"
              buttons
            ></b-form-radio-group>
          </b-form-group>
          <b-form-group id="ig-dep_st" label="DEP ST" label-for="i-dep_st">
            <b-form-radio-group
              id="i-dep_st"
              v-model="form.dep_st"
              :options="booleanOptions"
              buttons
            ></b-form-radio-group>
          </b-form-group>
          <hr />
          <b-button type="submit" variant="primary" :disabled="invalid"
            >Submit</b-button
          >
          <!--            <b-button type="reset" variant="danger">Reset</b-button>-->
        </b-form>
      </b-col>

      <b-col md="8" v-if="explanation" class="position-relative">
        <b-card no-body>
          <b-card-body>
            <h3>
              Patient risk:
              <b-badge :variant="explanation.risk ? 'danger' : 'success'">
                {{ explanation.risk ? "High" : "Low" }}</b-badge
              >
            </h3>
          </b-card-body>
          <b-card-body v-if="explanation.explanation.rule.premise">
            <h4>
              Why the risk is
              <b-badge :variant="explanation.risk ? 'danger' : 'success'">
                {{ explanation.risk ? "High" : "Low" }}</b-badge
              >?
            </h4>
            <b-list-group>
              <b-list-group-item
                v-for="r in explanation.explanation.rule.premise"
                :variant="explanation.risk ? 'danger' : 'success'"
              >
                <b>{{ r.att }} {{ r.op }} {{ +r.thr.toFixed(3) }}</b>
              </b-list-group-item>
            </b-list-group>
          </b-card-body>
          <b-card-body v-if="explanation.explanation.crules">
            <h4>
              The risk would have been
              <b-badge :variant="!explanation.risk ? 'danger' : 'success'">
                {{ !explanation.risk ? "High" : "Low" }}</b-badge
              >
              if:
            </h4>
            <b-list-group v-for="crule in explanation.explanation.deltas">
              <b-list-group-item v-for="cr in crule" variant="warning">
                <b>{{ cr.att }} {{ cr.op }} {{ +cr.thr.toFixed(3) }}</b>
              </b-list-group-item>
            </b-list-group>
          </b-card-body>
        </b-card>
        <b-overlay no-wrap :show="invalid"></b-overlay>
      </b-col>
    </b-row>
  </b-container>
</template>

<script>
import { getSingleEndpoint } from "@/axiosInstance";

export default {
  name: "PatientClassifier",
  data() {
    return {
      form: {
        age: 88,
        hr: 108,
        sbp: 117,
        creat: 4,
        killip: 4,
        caa: 0,
        tn: 1,
        dep_st: 0,
      },
      killipOptions: [
        { text: "class 1", value: 1 },
        { text: "class 2", value: 2 },
        { text: "class 3", value: 3 },
        { text: "class 4", value: 4 },
      ],
      booleanOptions: [
        { text: "yes", value: 1 },
        { text: "no", value: 0 },
      ],
      txtExplanation:
        "r = { Creat > 3.94, Age > 76.50, SBP <= 138.00, HR > 74.50 } --> " +
        "{ Risk: True }\nc = { { Creat <= 1.86 },\n      { Age <= 69.50 } }\n",
      explanation: null,
      invalid: false,
      responses: {
        q1: 0,
        q2: 0,
        q3: 0,
        n1: "",
        n2: "",
        n3: "",
      },
    };
  },
  methods: {
    onSubmit(event) {
      event.preventDefault();
      this.invalid = true;
      getSingleEndpoint(this.form, "explain_patient").then((res) => {
        this.explanation = res.data;
        this.invalid = false;
      });
      // console.log(this.form);
    },
    range(start, end) {
      const length = end - start;
      return Array.from({ length }, (_, i) => start + i);
    },
  },
};
</script>

<style scoped></style>
