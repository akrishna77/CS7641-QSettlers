/*
 * Java Settlers - An online multiplayer version of the game Settlers of Catan
 * This file Copyright (C) 2017-2019 Jeremy D Monin <jeremy@nand.net>
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
 */
package soc.robot.sample3p;

import soc.game.SOCGame;
import soc.game.SOCGameOption;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;
import soc.game.SOCTradeOffer;
import soc.message.SOCMessage;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.robot.SOCRobotNegotiator;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

import java.io.DataInputStream;
import java.io.DataOutputStream;

import java.net.Socket;

/**
 * Sample of a trivially simple "third-party" subclass of {@link SOCRobotBrain}
 * Instantiated by {@link Sample3PClient}.
 *<P>
 * Trivial behavioral changes from standard {@code SOCRobotBrain}:
 *<UL>
 * <LI> When sitting down, greet the game members: {@link #setOurPlayerData()}
 * <LI> Reject trades unless we're offered clay or sheep: {@link #considerOffer(SOCTradeOffer)}
 *</UL>
 *
 * @author Jeremy D Monin
 * @since 2.0.00
 */
public class Sample3PBrain extends SOCRobotBrain
{
    /**
     * Standard brain constructor; for javadocs see
     * {@link SOCRobotBrain#SOCRobotBrain(SOCRobotClient, SOCRobotParameters, SOCGame, CappedQueue)}.
     */

    private Socket servercon;
    private DataInputStream serverin;
    private DataOutputStream serverout;


    public Sample3PBrain(SOCRobotClient rc, SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq)
    {
        super(rc, params, ga, mq);
    }

    /**
     * {@inheritDoc}
     *<P>
     * After the standard actions of {@link SOCRobotBrain#setOurPlayerData()},
     * sends a "hello" chat message as a sample action using {@link SOCRobotClient#sendText(SOCGame, String)}.
     *<P>
     * If the for-bots extra game option {@link SOCGameOption#K__EXT_BOT} was set at the server command line,
     * prints its value to {@link System#err}. A third-party bot might want to use that option's value
     * to configure its behavior or debug settings.
     *<P>
     *<B>I18N Note:</B> Robots don't know what languages or locales the human players can read:
     * It would be unfair for a bot to ever send text that the players must understand
     * for gameplay. So this sample bot's "hello" is not localized.
     */
    @Override
    public void setOurPlayerData()
    {
        /*
        try{
            servercon = new Socket("localhost", 2004);
            servercon.setSoTimeout(300000);
            serverin = new DataInputStream(servercon.getInputStream());
            serverout = new DataOutputStream(servercon.getOutputStream());
            serverout.writeUTF("I just set player's data?");
            serverout.flush();
            serverout.close();  
            servercon.close();
            }
         catch(Exception e){
            System.err.println("Whoops!");
         }
         */   

        super.setOurPlayerData();

        final String botName = client.getNickname();
        client.sendText(game, "Hello from sample bot " + botName + "!");

        final String optExtBot = game.getGameOptionStringValue(SOCGameOption.K__EXT_BOT);
        if (optExtBot != null)
            System.err.println("Bot " + botName + ": __EXT_BOT is: " + optExtBot);
    }

    /**
     * Consider a trade offer; reject if we aren't offered clay or sheep.
     *<P>
     * {@inheritDoc}
     */
    @Override
    protected int considerOffer(SOCTradeOffer offer)
    {
        
        if (! offer.getTo()[getOurPlayerNumber()])
        {
            return SOCRobotNegotiator.IGNORE_OFFER;
        }

        String answer = "0";

        try{
            String giveData = "";
            SOCResourceSet give = offer.getGiveSet();
            giveData += Integer.toString(give.getAmount(SOCResourceConstants.CLAY));
            giveData += ",";
            giveData += Integer.toString(give.getAmount(SOCResourceConstants.WOOD));
            giveData += ",";
            giveData += Integer.toString(give.getAmount(SOCResourceConstants.SHEEP));
            giveData += ",";
            giveData += Integer.toString(give.getAmount(SOCResourceConstants.ORE));
            giveData += ",";
            giveData += Integer.toString(give.getAmount(SOCResourceConstants.WHEAT));

            String getData = "";
            SOCResourceSet get = offer.getGetSet();
            getData += Integer.toString(get.getAmount(SOCResourceConstants.CLAY));
            getData += ",";
            getData += Integer.toString(get.getAmount(SOCResourceConstants.WOOD));
            getData += ",";
            getData += Integer.toString(get.getAmount(SOCResourceConstants.SHEEP));
            getData += ",";
            getData += Integer.toString(get.getAmount(SOCResourceConstants.ORE));
            getData += ",";
            getData += Integer.toString(get.getAmount(SOCResourceConstants.WHEAT));

            servercon = new Socket("localhost", 2004);
            servercon.setSoTimeout(300000);
            serverin = new DataInputStream(servercon.getInputStream());
            serverout = new DataOutputStream(servercon.getOutputStream());
            String msg = "trade|10|" + getData + "|" + giveData;
            serverout.writeUTF(msg);
            /*
            while ((answer = serverin.readLine()) != null) {
                 System.err.println(answer);
            }
            */
 
            serverout.flush();
            serverout.close(); 
            serverin.close();   
            servercon.close();  
            }
         catch(Exception e){
            System.err.println("Whoops!");
         }

        final SOCResourceSet res = offer.getGiveSet();
        if (! (res.contains(SOCResourceConstants.CLAY) || res.contains(SOCResourceConstants.SHEEP)))
        {
            return SOCRobotNegotiator.REJECT_OFFER;
        }

        return super.considerOffer(offer);
    }
}
