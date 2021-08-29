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
#include <l1.h>

#define MAX_SAMPLES 10000000

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
  int samples = 0;

  if (av[1] == NULL)
    usage(av[0]);
  samples = atoi(av[1]);
  if (samples < 0)
    usage(av[0]);
  if (samples > MAX_SAMPLES)
    samples = MAX_SAMPLES;
  l1pp_t l1 = l1_prepare();

  int nsets = l1_getmonitoredset(l1, NULL, 0);

  int *map = calloc(nsets, sizeof(int));
  l1_getmonitoredset(l1, map, nsets);

  int rmap[L1_SETS];
  for (int i = 0; i < L1_SETS; i++)
    rmap[i] = -1;
  for (int i = 0; i < nsets; i++)
    rmap[map[i]] = i;


  uint16_t *res = calloc(samples * nsets, sizeof(uint16_t));
  for (int i = 0; i < samples * nsets; i+= 4096/sizeof(uint16_t))
    res[i] = 1;

  float *avg = calloc(L1_SETS, sizeof(float));

  //delayloop(3000000000U);
  l1_repeatedprobe(l1, samples, res, 0);



  for (int i = 0; i < samples; i++) {
    for (int j = 0; j < L1_SETS; j++) {
      if (rmap[j] == -1)
	       printf("  0 ");
      else{
	       //printf("%3d ", res[i*nsets + rmap[j]]);
         avg[rmap[j]] += res[i*nsets + rmap[j]];
      }
    }
    //putchar('\n');
  }


  for (int i = 0; i < L1_SETS; i++) {
    printf("%3f1 ", avg[rmap[i]] / samples);
  }
  putchar('\n');

  printf("Sorted: ");
  qsort(avg, L1_SETS, sizeof(float), flcompare);
  for (int i = 0; i < L1_SETS; i++) {
    printf("%3f1 ", avg[i] / samples);
  }
  putchar('\n');

  free(map);
  free(res);
  l1_release(l1);
}
