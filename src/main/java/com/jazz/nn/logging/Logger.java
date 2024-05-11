package com.jazz.nn.logging;

import com.jazz.nn.logging.utils.LogEntry;
import com.jazz.nn.logging.utils.LogLevel;

import java.util.ArrayList;
import java.util.List;

import static com.jazz.nn.logging.utils.LogLevel.*;

public class Logger {
    private LogLevel logLevel = ALL;

    private List<LogEntry> history;

    public Logger() {
        this.history = new ArrayList<>();
    }

    public void setLogLevel(LogLevel logLevel) {
        this.logLevel = logLevel;
    }

    public void log(LogLevel level, String text){
        history.add(new LogEntry(level,text));
        if(level.ordinal() >= logLevel.ordinal()){
            System.out.printf("[%s] %s%n", level.name(), text);
        }
    }

    public void info(String text){
        log(INFO,text);
    }

    public void debug(String text){
        log(DEBUG,text);
    }

    public void debug(String prefix, String text){
        history.add(new LogEntry(DEBUG,text));
        if(DEBUG.ordinal() >= logLevel.ordinal()){
            System.out.printf("[%s] %s %s%n", DEBUG.name(), prefix, text);
        }
    }

    public void error(String text){
        log(ERROR,text);
    }

    public List<String> get(LogLevel onlyAtLevel){
        List<String> strings = new ArrayList<>();
        for (LogEntry entry : history) {
            if(entry.getLogLevel().ordinal() == onlyAtLevel.ordinal()){
                strings.add(entry.getText());
            }
        }
        return strings;
    }
}
