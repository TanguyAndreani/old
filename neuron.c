#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct Node {
    double *weights;
    double net_input;
    double output;
};

struct Layer {
    int start; // 1 if this is the input layer.
    int n; // number of nodes in the layer
    struct Node *nodes;
    struct Layer *child;
    int b; // number of biases
    struct Node *biases;
};

double e = 2.71828;

void init_layer(struct Layer *l, int n, int b) {
  int i;
  l->n = n;
  l->b = b;
  l->start = 0;
  l->child = NULL;
  l->nodes = malloc(l->n * sizeof(struct Node));
  l->biases = malloc(l->b * sizeof(struct Node));
  for (i = 0; i < l->n; i++) {
    (l->nodes[i]).net_input = 0;
    (l->nodes[i]).output = 0;
  }
}

void init_weights(struct Layer *l) {
  if (l->child != NULL) {
    init_weights(l->child);
  } else {
    return;
  }
  int i = 0, k = 0;
  for (i = 0; i < l->n; i++) {
    (l->nodes[i]).weights = malloc(l->child->n * sizeof(double));
    for (k = 0; k < l->child->n; k++) {
      (l->nodes[i]).weights[k] = (double)rand()/RAND_MAX*2.0-1.0;
      //printf("Initializing some weights (%f)...\n", (l->nodes[i]).weights[k]);
    }
  }
  for (i = 0; i < l->b; i++) {
    (l->biases[i]).weights = malloc(l->child->n * sizeof(double));
    for (k = 0; k < l->child->n; k++) {
      (l->biases[i]).weights[k] = (double)rand()/RAND_MAX*2.0-1.0;
    }
  }
}

double f(double x) {
  return 1/(1 + pow(e, -x));
}

double *final_output(struct Layer *l) {
  int j = 0, k = 0;
  double sum = 0;
  double *z;
  if (l->child == NULL) {
    // output layer
    //printf("Getting into the output layer!\n");
    z = malloc(l->n * sizeof(double));
    for (k = 0; k < l->n; k++) {
      z[k] = f((l->nodes[k]).net_input);
    }
    return z;
  } else {
    // input layer
    //printf("Getting into some layer...\n");
    for (j = 0; j < l->child->n; j++) {
      sum = 0;
      //printf("Computing the net_input for each nodes (%d) of the next layer...\n", j);
      for (k = 0; k < l->n; k++) {
        if (l->start != 1) {
          (l->nodes[k]).output = f((l->nodes[k]).net_input);
        }
        sum += (l->nodes[k]).weights[j] * (l->nodes[k]).output;
      }
      for (k = 0; k < l->b; k++) {
        sum += (l->biases[k]).weights[j];
      }
      //printf("Net input of the %d'th of the next layer is %f.\n", j, sum);
      (l->child->nodes[j]).net_input = sum;
    }
    final_output(l->child);
  }
}

// n is the number of output nodes
double total_error(double *z, double *o, int n) {
  int k = 0;
  double te = 0;
  for (k = 0; k < n; k++) {
    te += pow(o[k] - z[k], 2);
  }
  te = te / n;
  return te;
}

void update_weights(struct Layer *root, struct Layer *l, double tres, double *o, int n) {
  if (l->child != NULL) {
    update_weights(root, l->child, tres, o, n);
  } else {
    return;
  }
  double te1, te2, *z;
  int i = 0, k = 0;
  for (i = 0; i < l->n; i++) {
    for (k = 0; k < l->child->n; k++) {
      (l->nodes[i]).weights[k] += tres;
      z = final_output(root);
      te1 = total_error(z, o, n);
      free(z);
      (l->nodes[i]).weights[k] -= 2*tres;
      z = final_output(root);
      te2 = total_error(z, o, n);
      free(z);
      (l->nodes[i]).weights[k] += tres;
      (l->nodes[i]).weights[k] -= tres * ((te1 - te2)/(2*tres));
      //printf("Updating some weights (%f)...\n", (l->nodes[i]).weights[k]);
    }
  }
  for (i = 0; i < l->b; i++) {
    for (k = 0; k < l->child->n; k++) {
      (l->biases[i]).weights[k] += tres;
      te1 = total_error(final_output(root), o, n);
      (l->biases[i]).weights[k] -= 2*tres;
      te2 = total_error(final_output(root), o, n);
      (l->biases[i]).weights[k] += tres;
      (l->biases[i]).weights[k] -= tres * ((te1 - te2)/(2*tres));
      //printf("Updating some weights (%f)...\n", (l->biases[i]).weights[k]);
    }
  }
}

void load_inputs(struct Layer *l, double *inputs) {
  int k = 0;
  for (k = 0; k < l->n; k++) {
    (l->nodes[k]).output = inputs[k];
  }
}

void clean_network(struct Layer *l) {
  int i = 0, k = 0;
  if (l->child != NULL) {
    clean_network(l->child);
    for (i = 0; i < l->n; i++) {
      free((l->nodes[i]).weights);
    }
  }
  free(l->nodes);
}

int main(void) {
  int i = 0, k = 0, j = 0;
  struct Layer input_layer, hidden_layer, output_layer;
  double inputs[][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  double outputs[][1] = {{0}, {1}, {1}, {0}};
  double *z;
  double te;
  srand(time(NULL));
  init_layer(&input_layer, 7, 2);
  init_layer(&hidden_layer, 7, 2);
  init_layer(&output_layer, 1, 0);
  input_layer.child = &hidden_layer;
  hidden_layer.child = &output_layer;
  input_layer.start = 1;
  init_weights(&input_layer);
  do {
    for (i = 0; i < 4; i++, j++) {
      load_inputs(&input_layer, inputs[i]);
      z = final_output(&input_layer);
      te = total_error(z, outputs[i], output_layer.n);
      printf("te = %f\r", te);
      free(z);
      update_weights(&input_layer, &input_layer, 0.1, outputs[i], output_layer.n);
    }
  } while (te >= 0.1);
  printf("\nCompleted after %d passes.\n", j);
  clean_network(&input_layer);
} 