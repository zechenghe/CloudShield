/*
 * Copyright 2016 CSIRO
 *
 * This file is part of Mastik.
 *
 * Mastik is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Mastik is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Mastik.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <util.h>
#include <l3.h>

#define MAX_SAMPLES 100000000

void usage(const char *prog) {
  fprintf(stderr, "Usage: %s <samples>\n", prog);
  exit(1);
}

int flcompare(const void *p1, const void *p2) {
  float u1 = *(float *)p1;
  float u2 = *(float *)p2;

  return u1 - u2;
}

int main(int ac, char **av) {
  delayloop(3000000000U);

  if (av[1] == NULL)
    usage(av[0]);
  int samples = atoi(av[1]);
  if (samples < 0)
    usage(av[0]);
  if (samples > MAX_SAMPLES)
    samples = MAX_SAMPLES;

  l3pp_t l3 = l3_prepare(NULL);
  int nsets = l3_getSets(l3);

  printf("nsets %d \n", nsets);

  int nmonitored = nsets/64;

  for (int i = 17; i < nsets; i += 64)
    l3_monitor(l3, i);


  uint16_t *res = calloc(samples * nmonitored, sizeof(uint16_t));
  for (int i = 0; i < samples * nmonitored; i+= 4096/sizeof(uint16_t))
    res[i] = 1;

  float *avg = calloc(nmonitored, sizeof(float));

  printf("Attack starts\n");

  int loops = 0;
  while(1){
    if (loops % 1000 == 0){
      printf("Probe loops %d \n", loops);
    }
    l3_repeatedprobecount(l3, samples, res, 0);
    loops += 1;
  }
/*
  for (int i = 0; i < samples; i++) {
    for (int j = 0; j < nmonitored; j++) {
      printf("%4d ", res[i*nmonitored + j]);
    }
    putchar('\n');
  }
*/

//  for (int i = 0; i < nmonitored; i++) {
//    printf("%3f1 ", avg[i] / samples);
//  }
//  putchar('\n');

  printf("Attack stops\n");
  free(res);
  l3_release(l3);
}
