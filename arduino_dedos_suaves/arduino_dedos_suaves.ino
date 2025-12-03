#include <Servo.h>

Servo thumb, indexF, middle, ring, pinky;

// posições atuais dos servos
int posAtual[5] = {90, 90, 90, 90, 90};
// posições desejadas
int posDesejada[5] = {90, 90, 90, 90, 90};

void setup() {
  Serial.begin(9600);
  thumb.attach(7);
  indexF.attach(8);
  middle.attach(9);
  ring.attach(10);
  pinky.attach(11);

  // inicializa todos os servos na posição neutra
  thumb.write(90);
  indexF.write(90);
  middle.write(90);
  ring.write(90);
  pinky.write(90);
}

void loop() {

  // lê os dados do Python
  if (Serial.available()) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();

    int d1 = getValor(comando, "thumb");
    int d2 = getValor(comando, "index");
    int d3 = getValor(comando, "middle");
    int d4 = getValor(comando, "ring");
    int d5 = getValor(comando, "pinky");

    if (d1 != -1) posDesejada[0] = d1;
    if (d2 != -1) posDesejada[1] = d2;
    if (d3 != -1) posDesejada[2] = d3;
    if (d4 != -1) posDesejada[3] = d4;
    if (d5 != -1) posDesejada[4] = d5;
  }

  // suaviza o movimento (incremento progressivo)
  moverSuave(thumb, 0);
  moverSuave(indexF, 1);
  moverSuave(middle, 2);
  moverSuave(ring, 3);
  moverSuave(pinky, 4);

  delay(10); // ajuste de suavidade
}

void moverSuave(Servo &servo, int idx) {
  if (posAtual[idx] == posDesejada[idx]) return;

  if (posAtual[idx] < posDesejada[idx]) posAtual[idx]++;
  else posAtual[idx]--;

  servo.write(posAtual[idx]);
}

int getValor(String data, String chave) {
  int pos = data.indexOf(chave + "=");
  if (pos == -1) return -1;
  int start = pos + chave.length() + 1;
  int end = data.indexOf(';', start);
  if (end == -1) end = data.length();
  return data.substring(start, end).toInt();
}
