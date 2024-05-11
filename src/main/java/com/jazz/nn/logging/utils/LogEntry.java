package com.jazz.nn.logging.utils;

public class LogEntry {
    private LogLevel logLevel;
    private String text;

    public LogEntry(LogLevel logLevel, String text) {
        this.logLevel = logLevel;
        this.text = text;
    }

    public LogLevel getLogLevel() {
        return logLevel;
    }

    public String getText() {
        return text;
    }
}
