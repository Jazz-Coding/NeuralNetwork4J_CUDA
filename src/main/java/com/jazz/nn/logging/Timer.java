package com.jazz.nn.logging;

public class Timer {
    private String label;
    private long start;
    private long end;
    private long duration;
    private float durationMS;

    public Timer(String label) {
        this.label = label;
    }

    public void start(){
        start = System.nanoTime();
    }

    public void stop(){
        end = System.nanoTime();
        duration = end-start;
        durationMS = duration/1e6F;
    }

    public void report(Logger logger){
        logger.debug(String.format("<%s> Runtime: %.2fms", label, durationMS));
    }
}
