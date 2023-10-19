##### Rebeca Maestro López

# INFORME ARQUITECTURA ULTRASPARC

La arquitectura UltraSPARC- I pertenece a la familia de procesadores SPARC (Arquitecturas de Procesadores Escalables). Por ello esta arquitectura funciona en un amplio rango desde microordenadores hasta superordenadores.

UltraSPARC- I es un procesador altamente integrado implementado con la arquitectura SPARC V9  64-bit RISC (lanzada en 1993).

Se especializa en servidores y estaciones de trabajo escalables.

### Características

* Multi-Processing Support

* Glueless 4-processor connection with minimum latency

* Snooping or Directory Based Protocol Support

##### Implementación

* 16KB cache de instrucciones

* Unidad de ejecución con dos ALUs

* Load buffer

* Store buffer

* Data cache (Primer nivel de cache)

* Data Memory Management Unit (DMMU)

* External cache (Segundo nivel de cache)

#### Multiprocesadores

Como implementa la arquitectura SPARC V9 64-bit RISC esta explota TLP de grano fino, también ILP aunque no tiene SMT.

Tiene un sistema SMP (Shared memory multiprocessor),con gran ancho de banda, además la conexión fisica entre el procesador y la interfaz del bus de datos consiste en un bus de direcciones.

### Coherencia

Soporta multiprocesamiento, ya que implementa conexiones de 4 procesadores con minima latencia y soporta tanto la coherencia por sondeo como la coherencia por directorio .

Utiliza la política de invalidación por escritura, asegurandose que un procesador tiene acceso exclusivo a un dato antes de que acceda a él y invalidando el resto de copias.

El protocolo usado para mantener la coherencia entre las caches internas , externas y el sistema es de tipo point-to-point write invalidate y esta basado en el protocolo MOESI de invalidación de 5 estados.

Usando los mismos tags para los estados:

* Exclusivo modificado (M)

* Compartido modificado (O)

* Exclusivo limpio (E)

* Compartido limpio (S)

* Invalido (I)
  
  La unidad de coherencia de cache es un bloque de tamaño 64 bytes y las transacciones de coherencia en lectura y escritura transfieren los datos en bloques de 64 byte.

### Consistencia :

Soporta los tres modelos de memoria de consistencia secuencial

- Total Store Order

- Partial Store Order

- Relaxed Memory Order
  
  Aunque la arquitectura afirma que no se basa en estos si no en las instrucciones MEMBAR y un modelo más débil que conduce a una mejor performance.

### Sincronización.

Tiene instrucciones de sincronización 

Principalmente FLUSH y  MEMBAR (Barrera ambas)

- Lookaside Barrier
  
  operación atómica de lectura y luego ajuste de memoria

- Memory Issue Barrier
  
  compara el contenido de un registro con un valor en memoria e intercambia memoria con el contenido de otro registro si la comparación fue igual

- Synchronization Barrier 
  
  se utilizan para sincronizar el orden de la memoria compartida en operaciones observadas por los procesadores.

### Rendimiento global:

La nueva arquitectura contiene 16 regitros de doble precisión adicionales de coma flotante, lo que eleva el total a 32. Estos registros adicionales reducen el tráfico de memoria, lo que permite que los programas se ejecuten más rápido. 

Los nuevos registros de punto flotante también se pueden direccionar como ocho registros de precisión cuádruple. El soporte de SPARC-V9 para un formato de punto flotante cuádruple de 128 bits es exclusivo para microprocesadores.

##### BIBLIOGRAFÍA:

http://datasheets.chipdb.org/Sun/stp1030.pdf

https://www.cs.utexas.edu/users/novak/sparcv9.pdf

https://www.oracle.com/technetwork/server-storage/sun-sparc-enterprise/documentation/sparc-usersmanual-2516676.pdf


