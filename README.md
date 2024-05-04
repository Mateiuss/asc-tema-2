Nume: DUDU Matei-Ioan

Grupă: 333CB

# Tema 2

Implementare
-

* Pentru a avea o oarecare cursivitate in explicarea temei, voi incepe mai intai sa scriu despre ce am facut in main, dupa care in kernel.
* Am decis ca valoarea maxima pe care o poate avea nonce sa o impart in blocuri de cate 512 thread-uri
```c
int threadsPerBlock = 512;
int blocks = (int)(MAX_NONCE) / threadsPerBlock;

if ((int)(MAX_NONCE) % threadsPerBlock != 0) {
    blocks++;
}
```
* Pentru device copiez array-urile block_content si difficulty in doua array-uri echivalente alocate in VRAM
```c
BYTE *device_block_content;
cudaMalloc(&device_block_content, BLOCK_SIZE);
cudaMemcpy(device_block_content, block_content, BLOCK_SIZE, cudaMemcpyHostToDevice);

BYTE *difficulty;
cudaMalloc(&difficulty, SHA256_HASH_SIZE);
cudaMemcpy(difficulty, DIFFICULTY, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
```
* De asemenea, aloc pentru device si o zona de memorie pentru nonce (zona de memorie in care se va afla rezultatul dupa apelarea kernel-ului)
* Dupa apelarea kernel-ului, calculez hash-ul nonce-ului gasit (dupa modelul variantei pe CPU)
* Kernel-ul findNonce are urmatorul prototip:
```c
__global__ void findNonce(BYTE *difficulty, BYTE *block_content, size_t current_length, uint32_t *nonce);
```
* Pentru fiecare thread calculez id-ul acestuia + 1 (ca sa pot sa pornesc de la valoarea 1 asa cum este mentionat in enunt)
* Thread-ul va incheia executia functiei daca id-ul acestuia este mai mare decat nonce-ul maxim sau a fost gasit deja un rezultat favorabil:
```c
if (thread_id > MAX_NONCE || *nonce != 0) {
    return;
}
```
* Altfel, thread-ul va calcula hash-ul pe baza id-ului sau si va compara hash-ul rezultat cu difficulty-ul. Nonce-ul va fi actualizat in situatia in care hash-ul este satisfacator

```c
if (compare_hashes(block_hash, difficulty) <= 0) {
    *nonce = thread_id;
}
```

Rezultate obtinute
-
```
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.08
```
* Am observat faptul ca doar prin verificarea daca a fost gasit un nonce am trecut de la aproximativ 1.20 la 0.08 secunde
* Presupun ca nonce-urile care se incadreaza in dificultatea data sunt destul de bine distribuite si nu se afla in zone relativ apropiate unele de altele, asa fel incat sa se ajunga la ele destul de repede indiferent de ordinea in care sunt parcurse blocurile

Resurse utilizate
-

* Paginile de la laboratoarele de CUDA de pe OCW

Git
-
* Link către [repo-ul de git](https://github.com/Mateiuss/asc-tema-2)
