/**
 * Java Settlers - An online multiplayer version of the game Settlers of Catan
 * Copyright (C) 2003  Robert S. Thomas <thomas@infolab.northwestern.edu>
 * Portions of this file Copyright (C) 2010,2013-2014,2017 Jeremy D Monin <jeremy@nand.net>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * The maintainer of this program can be reached at jsettlers@nand.net
 **/
package soc.message;

import java.util.StringTokenizer;


/**
 * This message means that a player wants to sit down to play.
 * Although it's a game-specific message, it's handled by {@code SOCServer} instead of a {@code GameHandler}.
 *
 * @author Robert S. Thomas
 */
public class SOCSitDown extends SOCMessage
    implements SOCMessageForGame
{
    private static final long serialVersionUID = 1111L;  // last structural change v1.1.11

    /**
     * Name of game
     */
    private String game;

    /**
     * Nickname of player; ignored from client, can send "-" but not blank
     */
    private String nickname;

    /**
     * The seat number
     */
    private int playerNumber;

    /**
     * True if this player is a robot
     */
    private boolean robotFlag;

    /**
     * Create a SitDown message.
     *
     * @param ga  the name of the game
     * @param nk  nickname of the player; ignored from client, can send "-" but not blank
     * @param pn  the seat number
     * @param rf  true if this is a robot
     */
    public SOCSitDown(String ga, String nk, int pn, boolean rf)
    {
        messageType = SITDOWN;
        game = ga;
        nickname = nk;
        playerNumber = pn;
        robotFlag = rf;
    }

    /**
     * @return the name of the game
     */
    public String getGame()
    {
        return game;
    }

    /**
     * @return the nickname of the player; can be "-" but not blank when sent from client
     */
    public String getNickname()
    {
        return nickname;
    }

    /**
     * @return the seat number
     */
    public int getPlayerNumber()
    {
        return playerNumber;
    }

    /**
     * @return the value of the robot flag
     */
    public boolean isRobot()
    {
        return robotFlag;
    }

    /**
     * SITDOWN sep game sep2 nickname sep2 playerNumber sep2 robotFlag
     *
     * @return the command string
     */
    public String toCmd()
    {
        return toCmd(game, nickname, playerNumber, robotFlag);
    }

    /**
     * SITDOWN sep game sep2 nickname sep2 playerNumber sep2 robotFlag
     *
     * @param ga  the name of the game
     * @param nk  nickname of the player; ignored from client, can send "-" but not blank
     * @param pn  the seat number
     * @param rf  the value of the robot flag
     * @return the command string
     */
    public static String toCmd(String ga, String nk, int pn, boolean rf)
    {
        return SITDOWN + sep + ga + sep2 + nk + sep2 + pn + sep2 + rf;
    }

    /**
     * Parse the command String into a SitDown message
     *
     * @param s   the String to parse
     * @return    a SitDown message, or null if the data is garbled
     */
    public static SOCSitDown parseDataStr(String s)
    {
        String ga; // the game name
        String nk; // nickname of the player
        int pn; // the seat number
        boolean rf; // the value of the robot flag

        StringTokenizer st = new StringTokenizer(s, sep2);

        try
        {
            ga = st.nextToken();
            nk = st.nextToken();
            pn = Integer.parseInt(st.nextToken());
            rf = (Boolean.valueOf(st.nextToken())).booleanValue();
        }
        catch (Exception e)
        {
            return null;
        }

        return new SOCSitDown(ga, nk, pn, rf);
    }

    /**
     * @return a human readable form of the message
     */
    public String toString()
    {
        return "SOCSitDown:game=" + game + "|nickname=" + nickname + "|playerNumber=" + playerNumber + "|robotFlag=" + robotFlag;
    }
}
